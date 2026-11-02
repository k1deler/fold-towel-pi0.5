"""CPU transform/data-loader preflight for real data; no pretrained model allocation."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

from .config import ASSET_ID, create, dataset_factory, file_sha, read_settings, runtime_sources
from .dataset import FoldDataset
from .prepare import atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','workspace','settings','openpi-root','base-params'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();s=read_settings(a.settings);reports=a.workspace/'reports'
    cfg=create(a.settings,a.openpi_root,a.workspace,a.base_params)
    from openpi.training import data_loader
    from openpi import transforms
    import jax
    if any(x.platform!='cpu' for x in jax.devices()):
        raise RuntimeError('Preparation/preflight must run on CPU; use JAX_PLATFORMS=cpu')
    data_cfg=cfg.data.create(cfg.assets_dirs,cfg.model)
    if not data_cfg.norm_stats or not data_cfg.use_quantile_norm:raise ValueError('Train-only π0.5 quantiles missing')
    if not cfg.model.pi05 or not cfg.model.discrete_state_input or cfg.ema_decay is not None:
        raise ValueError('Wrong π0.5 LoRA configuration')
    ds=FoldDataset(a.data,'val',s['action_horizon'],reports/'split.json',s['prompt'])
    transformed=data_loader.transform_dataset(ds,data_cfg)
    sample=transformed[0]
    if sample['state'].shape!=(32,) or sample['actions'].shape!=(s['action_horizon'],32):
        raise ValueError('Padding did not produce π0.5 state/action specs')
    if sample['tokenized_prompt'].shape!=(200,):raise ValueError('π0.5 state/prompt tokenization mismatch')
    if bool(sample['image_mask']['base_0_rgb']):raise ValueError('Synthetic head camera must remain masked')
    if any(im.shape!=(224,224,3) for im in sample['image'].values()):raise ValueError('OpenPI resize mismatch')
    original=ds[0]
    encoded=transforms.compose(data_cfg.data_transforms.inputs)({k:v for k,v in original.items()})
    normalized=transforms.Normalize(data_cfg.norm_stats,use_quantiles=True)(encoded)
    restored=transforms.Unnormalize(data_cfg.norm_stats,use_quantiles=True)(normalized)
    absolute=transforms.compose(data_cfg.data_transforms.outputs)(restored)['actions']
    difference=float(np.max(np.abs(absolute-original['actions'])))
    if difference>2e-5:raise ValueError('Physical action transform inverse mismatch')
    # Check the real upstream factory consumes only our train split and restores
    # its original LeRobot factory after leaving this scoped adapter.
    old_factory=data_loader.create_torch_dataset
    with dataset_factory(a.data,reports/'split.json',s['action_horizon'],s['prompt']):
        local=data_loader.create_torch_dataset(data_cfg,s['action_horizon'],cfg.model)
        assert len(local.episodes)==40 and not set(e['name'] for e in local.episodes)&set(json.loads((reports/'split.json').read_text())['val'])
        loader=data_loader.create_data_loader(cfg,shuffle=True,num_batches=1,framework='pytorch')
        observation,actions=next(iter(loader))
        assert tuple(actions.shape)==(s['batch_size'],s['action_horizon'],32)
    if data_loader.create_torch_dataset is not old_factory:raise RuntimeError('Scoped factory was not restored')
    forbidden=('pyrealsense2','i2rt','portal')
    if any(name in sys.modules for name in forbidden):raise RuntimeError('Live robot dependency imported')
    import importlib.metadata
    result={'status':'cpu_preflight_passed','time_wall_ns':time.time_ns(),
            'openpi_expected_commit':s['openpi_commit'],'settings_sha256':file_sha(a.settings),
            'runtime_sources':runtime_sources(a.openpi_root),
            'environment':{'python':sys.executable,'jax':jax.__version__,
                           'flax':importlib.metadata.version('flax'),'numpy':np.__version__},
            'norm_stats_sha256':file_sha(cfg.assets_dirs/ASSET_ID/'norm_stats.json'),
            'real_dataset_batch_shape':list(actions.shape),'model_action_dim':32,'physical_action_dim':14,
            'prompt_tokens':200,'input_image_size':[224,224],
            'physical_action_roundtrip_max_error':difference,'train_episodes':40,'val_episodes':11,
            'base_weight_metadata_present':(a.base_params/'_METADATA').exists(),
            'parameter_restore_tested':False,'model_gradient_tested':False,'fine_tuning_started':False,
            'robot_commands_sent':0}
    atomic(reports/'preflight.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
