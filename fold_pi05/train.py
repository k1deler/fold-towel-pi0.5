"""Explicit π0.5 LoRA smoke/train commands; preparation never launches this module."""
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

from .config import ASSET_ID, create, dataset_factory, file_sha, read_settings, upstream_trainer, runtime_sources


def gpu_info(index):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True,timeout=10)
    rows=[[int(x.strip()) for x in line.split(',')] for line in raw.splitlines()]
    return next((dict(index=i,used_mib=u,total_mib=t,utilization=v) for i,u,t,v in rows if i==index),None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','workspace','settings','openpi-root','base-params'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=('smoke','train'),required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute:raise SystemExit('Actual GPU compute requires --execute; use preflight for CPU checks')
    if sys.platform!='linux':raise RuntimeError('Training is restricted to the authorized Linux server')
    s=read_settings(a.settings);reports=a.workspace/'reports'
    preflight=json.loads((reports/'preflight.json').read_text())
    if preflight.get('status')!='cpu_preflight_passed' or preflight['settings_sha256']!=file_sha(a.settings):
        raise ValueError('Missing/stale CPU preflight')
    if preflight['runtime_sources']!=runtime_sources(a.openpi_root):
        raise ValueError('Adapter or pinned OpenPI source changed after preflight')
    preparation=json.loads((reports/'data_preparation.json').read_text())
    if preflight['norm_stats_sha256']!=preparation['norm_stats_sha256']:
        raise ValueError('Normalization report mismatch')
    if file_sha(reports/'split.json')!=preparation['contract']['split_sha256']:
        raise ValueError('Episode split changed after normalization')
    for name,digest in preparation['contract']['data_sha256'].items():
        if file_sha(a.data/name)!=digest:raise ValueError('Dataset changed after preparation')
    if a.mode=='train':
        marker=json.loads((reports/'gpu_smoke_passed.json').read_text())
        if marker['settings_sha256']!=file_sha(a.settings):raise ValueError('GPU smoke check is stale')
    # Do not inherit JAX_PLATFORMS=cpu from the data preparation shell.
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    os.environ['JAX_PLATFORMS']='cuda'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    if not str(a.base_params.resolve()).endswith('/pi05_base/params'):
        raise ValueError('Base checkpoint must be the separately verified π0.5 base params')
    import fcntl
    locks=Path.home()/'.cache/k1deler_training';locks.mkdir(parents=True,exist_ok=True)
    handle=(locks/f'gpu_{a.gpu}.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=gpu_info(a.gpu)
    if gpu is None or gpu['used_mib']>=1000 or gpu['utilization']>=5:
        raise RuntimeError('GPU is not idle. No foreign job is stopped; wait and choose an idle GPU.')
    cfg=create(a.settings,a.openpi_root,a.workspace,a.base_params,
               steps_override=2 if a.mode=='smoke' else None,
               experiment_override='smoke_seed42' if a.mode=='smoke' else None)
    if file_sha(cfg.assets_dirs/ASSET_ID/'norm_stats.json')!=preflight['norm_stats_sha256']:
        raise ValueError('Training stats changed')
    if cfg.checkpoint_dir.exists():raise FileExistsError('Preserving existing experiment; choose a new explicit config')
    from .prepare import atomic
    import jax
    if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':
        raise RuntimeError('Expected one selected CUDA device')
    state={'phase':a.mode+'_starting','gpu':gpu,'settings_sha256':file_sha(a.settings),
           'openpi_commit':s['openpi_commit'],'checkpoint_dir':str(cfg.checkpoint_dir),
           'action_semantics':preparation['contract'],'robot_commands_sent':0,
           'environment_from_preflight':preflight['environment']}
    atomic(reports/(a.mode+'_status.json'),state)
    try:
        # OpenPI's init_logging configures handlers[0], so establish its handler
        # just as the original standalone entrypoint does before calling main.
        logging.basicConfig(level=logging.INFO)
        trainer=upstream_trainer(a.openpi_root)
        smoke_metrics=[]
        if a.mode=='smoke':
            import numpy as np
            original_step=trainer.train_step
            def inspect_metrics(info):
                metrics={key:float(np.asarray(value)) for key,value in info.items()}
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError('Nonfinite π0.5 optimizer-step metrics')
                smoke_metrics.append(metrics)
            def checked_step(*arguments,**keywords):
                result,info=original_step(*arguments,**keywords)
                jax.debug.callback(inspect_metrics,info,ordered=True)
                return result,info
            trainer.train_step=checked_step
        # Use the same scoped adapter for training and normalization/inference.
        with dataset_factory(a.data,reports/'split.json',s['action_horizon'],s['prompt']):
            trainer.main(cfg)
        state['phase']=a.mode+'_complete'
        atomic(reports/(a.mode+'_status.json'),state)
        if a.mode=='smoke':
            if len(smoke_metrics)!=2:
                raise RuntimeError('Smoke run did not confirm two finite optimizer-step results')
            atomic(reports/'gpu_smoke_metrics.json',smoke_metrics)
            checkpoints=[p for p in cfg.checkpoint_dir.iterdir() if p.is_dir() and p.name.isdigit() and (p/'params').exists()]
            if not checkpoints:raise RuntimeError('Smoke trainer returned without an inference params checkpoint')
            atomic(reports/'gpu_smoke_passed.json',{'status':'passed','settings_sha256':file_sha(a.settings),
                   'checkpoint':str(sorted(checkpoints,key=lambda p:int(p.name))[-1]),
                   'scope':'real π0.5 base initialization and two optimizer steps; not task success',
                   'robot_commands_sent':0})
        print('PI05_'+a.mode.upper()+'_COMPLETE',flush=True)
    except BaseException as e:
        state.update(phase=a.mode+'_incomplete',error=f'{type(e).__name__}: {e}')
        atomic(reports/(a.mode+'_status.json'),state)
        raise
    finally:handle.close()


if __name__=='__main__':main()
