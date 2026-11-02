"""Bounded idle-GPU queue -> actual π0.5 smoke -> fine-tune, no robot I/O.

Only signals subprocesses created by this pipeline. A busy GPU is never selected.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def idle_candidates(rows):
    return [row['index'] for row in rows if row['used_mib']<1000 and row['utilization']<5
            and row['total_mib']-row['used_mib']>=32768]


def inventory():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True,timeout=15)
    return [dict(index=i,used_mib=m,total_mib=t,utilization=u) for i,m,t,u in
            ([int(value.strip()) for value in line.split(',')] for line in text.splitlines())]


def atomic(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--wait-hours',type=float,default=6)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute or sys.platform!='linux':
        raise SystemExit('Actual queue/training requires Linux and --execute')
    if not 0<args.wait_hours<=24:
        raise ValueError('Waiting budget must be positive and <=24 hours')
    workspace=args.workspace.resolve();reports=workspace/'reports';reports.mkdir(exist_ok=True)
    import fcntl
    queue_lock=(workspace/'train_pipeline.lock').open('a')
    fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight=json.loads((reports/'preflight.json').read_text())
    if preflight.get('status')!='cpu_preflight_passed':
        raise ValueError('CPU preflight is not passed')
    state={'task':'fold_towel','model':'pi05','method':'lora','pid':os.getpid(),
           'code_root':str(ROOT),'robot_commands_sent':0,'formal_training_started':False,
           'created_wall_ns':time.time_ns(),'max_wait_hours':args.wait_hours}
    child=None

    def record(phase,**values):
        state.update(phase=phase,updated_wall_ns=time.time_ns(),**values)
        atomic(reports/'training_pipeline_status.json',state)
        print('PI05_STATUS',phase,flush=True)

    def interrupted(signum,frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        raise InterruptedError('Operator stopped this pipeline')

    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)

    def run_stage(mode,gpu,seconds):
        nonlocal child
        log_path=reports/(mode+'_console.log')
        if log_path.exists():
            raise FileExistsError('Prior stage log exists; inspect before starting another run: '+str(log_path))
        with log_path.open('x') as log:
            child=subprocess.Popen(['bash',str(ROOT/'scripts/gpu_experiment.sh'),mode,str(gpu)],
                    cwd=ROOT,env=dict(os.environ,PI05_WORKSPACE=str(workspace)),
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            record(mode+'_running',gpu=gpu,child_pid=child.pid,log=log_path.name,
                   formal_training_started=mode=='train')
            try:
                code=child.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
                raise TimeoutError('This pipeline stage exceeded time budget: '+mode)
            if code:
                raise RuntimeError(f'{mode} exited {code}; see {log_path}')

    def wait_for_gpu(deadline,phase):
        record(phase,child_pid=None,child_running=False)
        while time.monotonic()<deadline:
            rows=inventory()
            state['latest_gpu_inventory']=rows
            state['latest_gpu_check_wall_ns']=time.time_ns()
            atomic(reports/'training_pipeline_status.json',state)
            for candidate in idle_candidates(rows):
                locks=Path.home()/'.cache/k1deler_training';locks.mkdir(parents=True,exist_ok=True)
                lock_path=locks/f'gpu_{candidate}.lock'
                with lock_path.open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    time.sleep(3)
                    if candidate not in idle_candidates(inventory()):continue
                # train.py takes the same lock and rechecks idle before allocating.
                return candidate
            time.sleep(30)
        raise TimeoutError('No idle GPU within waiting budget; no foreign job interrupted')

    try:
        deadline=time.monotonic()+args.wait_hours*3600
        gpu=wait_for_gpu(deadline,'waiting_for_idle_gpu')
        state['selected_gpu']=gpu
        run_stage('smoke',gpu,3600)
        smoke=json.loads((reports/'gpu_smoke_passed.json').read_text())
        if smoke.get('status')!='passed':raise ValueError('Actual optimizer/checkpoint smoke marker missing')
        record('smoke_passed',smoke_checkpoint=smoke['checkpoint'])
        # CUDA context of the smoke process is gone. Prefer its card if still
        # available; otherwise wait for an idle card without moving running jobs.
        if gpu not in idle_candidates(inventory()):
            gpu=wait_for_gpu(time.monotonic()+args.wait_hours*3600,'waiting_for_formal_training_gpu')
        run_stage('train',gpu,86400)
        marker=json.loads((reports/'train_status.json').read_text())
        if marker.get('phase')!='train_complete':raise RuntimeError('Fine-tune returned without completion report')
        record('fine_tuning_complete_offline',checkpoint_dir=marker['checkpoint_dir'],
               note='Optimizer steps/checkpoints completed; held-out evaluation and robot testing are separate.')
    except BaseException as exc:
        record('failed_or_incomplete',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        queue_lock.close()


if __name__=='__main__':main()
