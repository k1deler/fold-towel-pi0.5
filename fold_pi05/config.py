"""Pinned OpenPI configuration without editing its global registry or source."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from .dataset import FoldDataset, FoldInputs, FoldOutputs

EXPECTED_REVISION='215abfb217dbac7d5f1273282331b9b1866c0479'
REPO_ID='fold_towel_local_leader_v1'
ASSET_ID='yam_fold_towel'


def read_settings(path):
    settings=json.loads(Path(path).read_text(encoding='utf-8'))
    if settings['task']!='fold_towel' or settings['model']!='pi05' or settings['method']!='lora':
        raise ValueError('This initial candidate is a fold_towel π0.5 LoRA experiment')
    if settings['action_offset']!=0 or settings['openpi_commit']!=EXPECTED_REVISION:
        raise ValueError('Temporal/source contract changed')
    if (settings['action_dim'],settings['physical_action_dim'])!=(32,14):
        raise ValueError('Wrong padded/physical action dimensions')
    if not 1 <= settings['action_horizon'] <= 100:
        raise ValueError('Invalid action horizon')
    if not 0 < settings['warmup_steps'] < settings['steps']:
        raise ValueError('Invalid training schedule')
    if settings['batch_size']<1 or settings['num_workers']<0 or settings['fsdp_devices']!=1:
        raise ValueError('This candidate requires one explicitly selected GPU')
    return settings


def activate_openpi(root):
    root=Path(root).resolve()
    if not (root/'src/openpi/training/config.py').is_file():
        raise ValueError('Pinned OpenPI checkout missing')
    sys.path[:0]=[str(root/'src'),str(root/'packages/openpi-client/src')]
    existing=sys.modules.get('openpi')
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(root):
        raise RuntimeError('A different OpenPI checkout was already imported')
    return root


def create(settings_path,openpi_root,output,base_params,steps_override=None,experiment_override=None):
    settings=read_settings(settings_path)
    activate_openpi(openpi_root)
    from openpi.training import config as tc, optimizer, weight_loaders
    from openpi.models.pi0_config import Pi0Config
    from openpi import transforms
    model=Pi0Config(pi05=True,action_horizon=settings['action_horizon'],
                    paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
    data=tc.SimpleDataConfig(repo_id=REPO_ID,assets=tc.AssetsConfig(asset_id=ASSET_ID),
         data_transforms=lambda _:transforms.Group(inputs=[FoldInputs()],outputs=[FoldOutputs()]))
    steps=settings['steps'] if steps_override is None else steps_override
    if steps<1:raise ValueError('Invalid step count')
    warmup=min(settings['warmup_steps'],max(1,steps//2))
    return tc.TrainConfig(name=settings['name'],exp_name=experiment_override or settings['experiment'],
        model=model,data=data,weight_loader=weight_loaders.CheckpointWeightLoader(str(base_params)),
        assets_base_dir=str(Path(output).resolve()/'assets'),
        checkpoint_base_dir=str(Path(output).resolve()/'checkpoints'),
        seed=settings['seed'],batch_size=settings['batch_size'],num_workers=settings['num_workers'],
        num_train_steps=steps,fsdp_devices=1,freeze_filter=model.get_freeze_filter(),ema_decay=None,
        save_interval=min(settings['save_interval'],max(1,steps-1)),keep_period=settings['save_interval'],
        log_interval=min(settings['log_interval'],max(1,steps-1)),wandb_enabled=False,
        overwrite=False,resume=False,
        policy_metadata={'task':'fold_towel','action_source':'recorded_leader_absolute_at_observation_plus_k',
          'joint_representation':'delta_from_one_follower_anchor','gripper_representation':'absolute_0_1',
          'action_horizon':settings['action_horizon'],'fps':30,'robot_tested':False},
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=warmup,peak_lr=settings['peak_lr'],
            decay_steps=max(steps,warmup+1),decay_lr=settings['peak_lr']/10))


@contextmanager
def dataset_factory(data_root,split_path,horizon,prompt):
    """Process-local adapter; upstream module is restored even after an error."""
    from openpi.training import data_loader
    original=data_loader.create_torch_dataset
    def create_dataset(data_config,action_horizon,model_config):
        if data_config.repo_id==REPO_ID:
            if action_horizon!=horizon:
                raise ValueError('Model/dataset horizon mismatch')
            return FoldDataset(data_root,'train',horizon,split_path,prompt)
        return original(data_config,action_horizon,model_config)
    data_loader.create_torch_dataset=create_dataset
    try:yield
    finally:data_loader.create_torch_dataset=original


def upstream_trainer(openpi_root):
    path=Path(openpi_root)/'scripts/train.py'
    spec=importlib.util.spec_from_file_location('fold_pi05_openpi_train',path)
    if spec is None or spec.loader is None:raise RuntimeError('OpenPI trainer missing')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):result.update(data)
    return result.hexdigest()


def runtime_sources(openpi_root):
    """Fingerprint this adapter and actual upstream Python used by the process."""
    adapter=Path(__file__).resolve().parent
    files={f'adapter/{p.name}':file_sha(p) for p in sorted(adapter.glob('*.py'))}
    upstream=Path(openpi_root).resolve()
    for prefix in ('src/openpi','scripts','packages/openpi-client/src'):
        for p in sorted((upstream/prefix).rglob('*.py')):
            if '__pycache__' not in p.parts:
                files['openpi/'+p.relative_to(upstream).as_posix()]=file_sha(p)
    return files
