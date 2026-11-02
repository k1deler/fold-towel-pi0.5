"""Held-out offline evaluation AFTER real fine-tuning; never connects hardware."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from .config import create, read_settings
from .dataset import FoldDataset, GRIPPERS, JOINTS
from .prepare import atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','workspace','settings','openpi-root','base-params','checkpoint'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--samples',type=int,default=256)
    a=p.parse_args()
    if a.samples<1 or not (a.checkpoint/'params').is_dir():
        raise ValueError('Requires an actual fine-tuned OpenPI step directory and positive sample count')
    s=read_settings(a.settings)
    # This module must be called with explicitly selected idle GPU by its operator.
    import jax
    if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':
        raise RuntimeError('Select one idle CUDA GPU for offline evaluation')
    cfg=create(a.settings,a.openpi_root,a.workspace,a.base_params)
    from openpi.policies.policy_config import create_trained_policy
    policy=create_trained_policy(cfg,a.checkpoint)
    ds=FoldDataset(a.data,'val',s['action_horizon'],a.workspace/'reports/split.json',s['prompt'])
    ids=np.random.default_rng(s['seed']).choice(len(ds),min(a.samples,len(ds)),replace=False)
    absolute=np.zeros(14);squared=np.zeros(14);first=np.zeros(14);hold=np.zeros(14)
    counts=0;out_of_range=0;gripper_values=0;times=[]
    for index in ids:
        sample=ds[int(index)]
        target=sample.pop('actions')  # Never feed labels to an inference request.
        anchor=sample['state'].copy();started=time.monotonic()
        action=np.asarray(policy.infer(sample)['actions'])
        times.append(time.monotonic()-started)
        if action.shape!=target.shape or not np.isfinite(action).all():
            raise ValueError('Invalid predicted physical absolute action shape/value')
        err=action-target
        absolute+=np.abs(err).sum(0);squared+=(err**2).sum(0);first+=np.abs(err[0])
        hold+=np.abs(anchor[None]-target).sum(0);counts+=len(target)
        grips=action[:,GRIPPERS]
        out_of_range+=int(((grips<0)|(grips>1)).sum());gripper_values+=grips.size
    result={'scope':'held-out open-loop physical leader target prediction; not folding success',
            'checkpoint':str(a.checkpoint),'samples':len(ids),'indices':ids.tolist(),
            'joint_mae_rad':float((absolute/counts)[JOINTS].mean()),
            'joint_rmse_rad':float(np.sqrt(squared[JOINTS].sum()/(counts*len(JOINTS)))),
            'gripper_mae':float((absolute/counts)[GRIPPERS].mean()),
            'first_step_joint_mae_rad':float((first/len(ids))[JOINTS].mean()),
            'hold_joint_mae_rad':float((hold/counts)[JOINTS].mean()),
            'raw_gripper_out_of_range_fraction':out_of_range/gripper_values,
            'median_inference_seconds':float(np.median(times)),
            'predictions_clipped_before_metrics':False,'robot_commands_sent':0,
            'comparison_note':'ACT has H100, pi05 H50; align physical target horizons before comparing aggregate MAE.'}
    destination=a.workspace/'reports'/('eval_'+a.checkpoint.name+'.json')
    if destination.exists():raise FileExistsError('Preserving previous evaluation report')
    atomic(destination,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
