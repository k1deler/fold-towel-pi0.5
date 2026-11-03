"""Evaluate saved pi05 milestones with explicit GPU resource policy; no robot I/O."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from train_pipeline import atomic, idle_candidates, inventory

ROOT = Path(__file__).resolve().parents[1]
STEPS = (9999, 2000, 4000, 6000, 8000)


def eligible_gpus(rows, share_with_headroom=False, budget_mib=32768, reserve_mib=8192):
    if not share_with_headroom:
        return idle_candidates(rows)
    # Explicit user-authorized sharing: preserve a separate reserve above our
    # allocator budget and prefer the lowest current compute load.
    candidates = [row for row in rows
                  if row['total_mib']-row['used_mib'] >= budget_mib+reserve_mib]
    candidates.sort(key=lambda row: (row['utilization'], -(row['total_mib']-row['used_mib'])))
    return [row['index'] for row in candidates]


def validate_result(result, checkpoint, samples):
    if Path(result['checkpoint']).resolve() != checkpoint.resolve():
        raise ValueError('Evaluation report refers to a different checkpoint')
    if result['samples'] != samples or len(set(result['indices'])) != samples:
        raise ValueError('Wrong evaluation sample count or duplicate windows')
    for key in ('joint_mae_rad', 'joint_rmse_rad', 'gripper_mae',
                'first_step_joint_mae_rad', 'hold_joint_mae_rad',
                'raw_gripper_out_of_range_fraction', 'median_inference_seconds'):
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError('Invalid evaluation metric: ' + key)
    if result['raw_gripper_out_of_range_fraction'] > 1:
        raise ValueError('Invalid gripper excursion fraction')
    if result['robot_commands_sent'] != 0 or result['predictions_clipped_before_metrics']:
        raise ValueError('Evaluation must be offline and retain raw excursions')


def comparison(reports):
    indices = reports[0]['indices']
    if any(r['indices'] != indices for r in reports):
        raise ValueError('Checkpoint comparison requires identical held-out windows')
    best = min(reports, key=lambda r: r['joint_mae_rad'])
    keys = ('checkpoint', 'samples', 'joint_mae_rad', 'gripper_mae',
            'first_step_joint_mae_rad', 'hold_joint_mae_rad',
            'raw_gripper_out_of_range_fraction', 'median_inference_seconds')
    return {
        'scope': 'checkpoint selection on shared held-out windows; not independent test or folding success',
        'ranked_results': [{k: r[k] for k in keys} for r in sorted(reports, key=lambda r: r['joint_mae_rad'])],
        'selected_checkpoint': best['checkpoint'],
        'beats_hold_baseline': best['joint_mae_rad'] < best['hold_joint_mae_rad'],
        'raw_gripper_excursions_require_deployment_handling': best['raw_gripper_out_of_range_fraction'] > 0,
        'robot_commands_sent': 0,
        'automatic_deployment': False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--wait-hours', type=float, default=12)
    p.add_argument('--share-with-headroom', action='store_true',
                   help='Explicitly permit an occupied GPU with sufficient free VRAM')
    p.add_argument('--gpu-budget-mib', type=int, default=32768)
    p.add_argument('--reserve-mib', type=int, default=8192)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute or sys.platform != 'linux':
        raise SystemExit('Offline GPU evaluation requires Linux and --execute')
    if not 0 < a.wait_hours <= 24 or not 1 <= a.samples <= 5702:
        raise ValueError('Invalid bounded waiting budget/sample count')
    if a.gpu_budget_mib < 16384 or a.reserve_mib < 8192:
        raise ValueError('Evaluation needs at least 16GiB allocator budget and 8GiB reserve')
    import fcntl
    r = a.workspace.resolve()
    reports = r/'reports'
    lock = (r/'evaluate_pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'robot_commands_sent': 0, 'steps': list(STEPS),
             'samples': a.samples, 'created_wall_ns': time.time_ns(), 'completed': [],
             'resource_policy': 'shared_headroom' if a.share_with_headroom else 'idle_only',
             'gpu_budget_mib': a.gpu_budget_mib, 'reserve_mib': a.reserve_mib}
    child = None

    def record(phase, **values):
        state.update(phase=phase, updated_wall_ns=time.time_ns(), **values)
        atomic(reports/'evaluation_pipeline_status.json', state)
        print('PI05_EVALUATION', phase, flush=True)

    def stop_child():
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)

    def interrupted(signum, frame):
        stop_child()
        raise InterruptedError('Operator stopped this evaluation queue')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        train = json.loads((reports/'train_status.json').read_text())
        if train['phase'] != 'train_complete':
            raise ValueError('Fine-tuning is not complete; preserving the training job')
        checkpoint_root = Path(train['checkpoint_dir']).resolve()
        if checkpoint_root != (r/'checkpoints/fold_towel_pi05_lora/seed42_v1').resolve():
            raise ValueError('Unexpected training experiment')
        sys.path.insert(0, str(ROOT))
        from fold_pi05.config import file_sha, runtime_sources
        preflight = json.loads((reports/'preflight.json').read_text())
        if preflight['runtime_sources'] != runtime_sources(r/'vendor/openpi'):
            raise ValueError('Evaluation adapter differs from verified training sources')
        if file_sha(ROOT/'configs/pi05_lora.json') != train['settings_sha256']:
            raise ValueError('Settings differ from completed training')
        stats = preflight['norm_stats_sha256']
        for step in STEPS:
            ckpt = checkpoint_root/str(step)
            if not (ckpt/'params').is_dir() or not (ckpt/'_CHECKPOINT_METADATA').is_file():
                raise ValueError('Incomplete checkpoint: '+str(ckpt))
            if file_sha(ckpt/'assets/yam_fold_towel/norm_stats.json') != stats:
                raise ValueError('Checkpoint normalization differs from training')
        done = []
        deadline = time.monotonic()+a.wait_hours*3600
        for step in STEPS:
            checkpoint = checkpoint_root/str(step)
            report_path = reports/f'eval_{step}.json'
            if not report_path.exists():
                record('waiting_for_gpu_headroom' if a.share_with_headroom else 'waiting_for_idle_gpu',
                       pending_step=step, gpu=None, child_pid=None)
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('No eligible GPU within bounded queue budget')
                    if shutil.disk_usage(r).free < 20*1024**3:
                        raise RuntimeError('Less than 20GiB disk headroom; no checkpoints removed')
                    rows = inventory()
                    state['gpu_inventory'] = rows
                    state['gpu_checked_wall_ns'] = time.time_ns()
                    atomic(reports/'evaluation_pipeline_status.json', state)
                    candidates = eligible_gpus(rows, a.share_with_headroom, a.gpu_budget_mib, a.reserve_mib)
                    gpu_lock = None
                    for gpu in candidates:
                        locks = Path.home()/'.cache/k1deler_training'
                        locks.mkdir(parents=True, exist_ok=True)
                        candidate_lock = (locks/f'gpu_{gpu}.lock').open('a')
                        try:
                            fcntl.flock(candidate_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            time.sleep(3)
                            checked_rows = inventory()
                            if gpu not in eligible_gpus(checked_rows, a.share_with_headroom,
                                                        a.gpu_budget_mib, a.reserve_mib):
                                candidate_lock.close()
                                continue
                            selected_row = next(row for row in checked_rows if row['index'] == gpu)
                            gpu_lock = candidate_lock
                            break
                        except BlockingIOError:
                            candidate_lock.close()
                    if gpu_lock is not None:
                        break
                    time.sleep(30)
                try:
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS='cuda,cpu',
                               XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONDONTWRITEBYTECODE='1',
                               XLA_PYTHON_CLIENT_MEM_FRACTION=str(a.gpu_budget_mib/selected_row['total_mib']),
                               XLA_PYTHON_CLIENT_ALLOCATOR='default',
                               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2',
                               PYTHONPATH=os.pathsep.join(map(str, [ROOT, r/'vendor/openpi/src',
                                                         r/'vendor/openpi/packages/openpi-client/src'])))
                    command = [sys.executable, '-u', '-m', 'fold_pi05.evaluate',
                               '--data', str(Path.home()/'output/personal/k1deler/fold_towel_20261101/converted'),
                               '--workspace', str(r), '--settings', str(ROOT/'configs/pi05_lora.json'),
                               '--openpi-root', str(r/'vendor/openpi'),
                               '--base-params', str(Path.home()/'output/data/pretrained/pi05_base/params'),
                               '--checkpoint', str(checkpoint), '--samples', str(a.samples)]
                    log_path = reports/f'eval_{step}_console.log'
                    with log_path.open('x') as log:
                        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                                 start_new_session=True)
                        record('evaluating', gpu=gpu, child_pid=child.pid, pending_step=step,
                               selected_gpu_snapshot=selected_row, log=log_path.name)
                        try:
                            code = child.wait(timeout=3600)
                        except subprocess.TimeoutExpired:
                            stop_child()
                            raise TimeoutError('Own evaluation exceeded one-hour stage limit')
                    if code:
                        raise RuntimeError(f'Evaluation {step} failed with exit {code}; inspect {log_path}')
                finally:
                    gpu_lock.close()
            result = json.loads(report_path.read_text())
            validate_result(result, checkpoint, a.samples)
            done.append(result)
            state['completed'].append(step)
            record('checkpoint_evaluated', child_pid=None)
        summary = comparison(done)
        atomic(reports/'checkpoint_comparison.json', summary)
        record('evaluation_complete_offline', selected_checkpoint=summary['selected_checkpoint'], gpu=None)
    except BaseException as exc:
        stop_child()
        record('failed_or_incomplete', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
