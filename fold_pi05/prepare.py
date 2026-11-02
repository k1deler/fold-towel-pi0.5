"""Read-only dataset verification and train-only stats; default has no GPU use."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from .config import ASSET_ID, create, file_sha, read_settings
from .dataset import FoldDataset, FoldInputs, split_names


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)


def prepare(data,workspace,settings_path,act_audit,act_config,openpi_root,base_params):
    data=Path(data).resolve();workspace=Path(workspace).resolve();settings=read_settings(settings_path)
    reports=workspace/'reports';reports.mkdir(parents=True,exist_ok=True)
    expected=json.loads(Path(act_audit).read_text())
    for name,digest in expected['dataset_sha256'].items():
        if file_sha(data/name)!=digest:raise ValueError('ACT-verified dataset changed: '+name)
    meta=json.loads((data/'meta.json').read_text())
    episodes=sorted([e for e in meta['episodes'] if e['task']=='fold_towel'],key=lambda e:e['name'])
    split=split_names(episodes,settings['train_ratio'],settings['split_seed'])
    import yaml
    baseline=yaml.safe_load(Path(act_config).read_text())
    if set(split['train'])!=set(baseline['train_episodes']) or set(split['val'])!=set(baseline['val_episodes']):
        raise ValueError('New split differs from completed ACT baseline')
    atomic(reports/'split.json',split)
    train=FoldDataset(data,'train',settings['action_horizon'],reports/'split.json',settings['prompt'])
    val=FoldDataset(data,'val',settings['action_horizon'],reports/'split.json',settings['prompt'])
    # Verify every converted numeric row, but decode only first/mid/last selected
    # samples here; original ACT conversion audit already checked all stream indexes.
    arrays=train.arrays()
    if not np.isfinite(arrays['qpos']).all() or not np.isfinite(arrays['action']).all():
        raise ValueError('Nonfinite source physical data')
    for ds in (train,val):
        for index in (0,len(ds)//2,len(ds)-1):
            sample=ds[index]
            assert sample['actions'].shape==(settings['action_horizon'],14)
    configuration=create(settings_path,openpi_root,workspace,base_params)
    from openpi.shared import normalize
    statistics={key:normalize.RunningStats() for key in ('state','actions')}
    transform=FoldInputs();dummy=np.zeros((1,1,3),dtype=np.uint8)
    for start in range(0,len(train),256):
        states=[];actions=[]
        for index in range(start,min(start+256,len(train))):
            state,absolute=train.numeric(index)
            value=transform({'state':state,'actions':absolute,'images':{'left':dummy,'right':dummy},'prompt':settings['prompt']})
            states.append(value['state']);actions.append(value['actions'])
        statistics['state'].update(np.stack(states))
        statistics['actions'].update(np.stack(actions))
    assets=configuration.assets_dirs/ASSET_ID
    normalize.save(assets,{k:v.get_statistics() for k,v in statistics.items()})
    start_pose=np.mean([arrays['qpos'][e['start']] for e in train.episodes],axis=0).tolist()
    contract={'task':'fold_towel','data_sha256':expected['dataset_sha256'],'settings_sha256':file_sha(settings_path),
              'split_sha256':file_sha(reports/'split.json'),'horizon':settings['action_horizon'],'offset':0,
              'absolute_labels':'recorded leader(t+k)','relative_dimensions':[0,1,2,3,4,5,7,8,9,10,11,12],
              'absolute_grippers':[6,13],'source_order':'left_joint0..5,left_gripper,right_joint0..5,right_gripper',
              'missing_base_camera':'zero RGB, image_mask false','normalization_fitted_on':'train only'}
    atomic(assets/'fold_contract.json',contract)
    result={'status':'prepared_no_training','task':'fold_towel','time_wall_ns':time.time_ns(),
            'train_episodes':len(train.episodes),'val_episodes':len(val.episodes),
            'train_windows':len(train),'val_windows':len(val),'horizon':settings['action_horizon'],
            'same_episode_split_as_ACT':True,'train_only_start_pose':start_pose,'contract':contract,
            'norm_stats_sha256':file_sha(assets/'norm_stats.json'),'training_started':False,
            'model_loaded':False,'robot_commands_sent':0,'action_labels_match_ACT_physical_targets':True,
            'representation_difference_from_ACT':'pi05 joint delta / absolute grippers; ACT all14 delta'}
    atomic(reports/'data_preparation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','workspace','settings','act-audit','act-config','openpi-root','base-params'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.data,a.workspace,a.settings,a.act_audit,a.act_config,a.openpi_root,a.base_params)


if __name__=='__main__':main()
