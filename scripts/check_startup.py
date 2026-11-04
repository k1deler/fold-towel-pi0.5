"""Explicit GPU check of both JIT signatures and Enter-like idle waits. No devices."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    if sys.platform != 'linux':
        raise SystemExit('Use the authorized Linux training host')
    rows = [[int(x.strip()) for x in line.split(',')] for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True).splitlines()]
    row = next(r for r in rows if r[0] == a.gpu)
    if row[2]-row[1] < 40960:
        raise SystemExit('Need allocator budget 32GiB plus 8GiB free reserve')
    import fcntl
    lock_dir = Path.home()/'.cache/k1deler_training'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir/f'gpu_{a.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), JAX_PLATFORMS='cuda,cpu',
                          XLA_PYTHON_CLIENT_PREALLOCATE='false',
                          XLA_PYTHON_CLIENT_MEM_FRACTION=str(32768/row[2]),
                          PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                          OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
        root = a.workspace.resolve()
        sys.path[:0] = [str(root/'code'), str(root/'vendor/openpi/src'),
                       str(root/'vendor/openpi/packages/openpi-client/src')]
        import numpy as np
        from deploy_pi05.client import RemotePolicy
        from deploy_pi05.common import CAMERAS
        from deploy_pi05.service import Engine, handle, warmup
        sample = np.load(root/'replay/observation.npz', allow_pickle=False)
        frames = {name: sample[name] for name in CAMERAS}
        engine = Engine(root)
        start = time.monotonic()
        warmup(engine, frames, sample['qpos'], sample['noise'])
        warm_seconds = time.monotonic()-start
        with socket.create_server(('127.0.0.1', 0)) as server:
            def serve():
                connection, _ = server.accept()
                handle(engine, connection)
            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            client = RemotePolicy('127.0.0.1:'+str(server.getsockname()[1]),
                                  jpeg_roundtrip=False, expected_model_id=engine.model_id)
            durations = []
            try:
                # Reproduce a long operator wait, then use the NORMAL (noise=None)
                # request signature, rather than another fixed-noise replay.
                time.sleep(11)
                for _ in range(3):
                    client.predict(frames, sample['qpos'], 'fold_towel')
                    durations.append(client.last_request_seconds)
            finally:
                client.close()
            thread.join(timeout=3)
            if thread.is_alive():
                raise RuntimeError('Client disconnect did not end handler')
        result = {'status': 'both_signatures_and_normal_network_path_passed',
                  'model_id': engine.model_id, 'gpu': a.gpu, 'gpu_at_start': row,
                  'warmup_seconds': warm_seconds, 'operator_idle_seconds': 11,
                  'normal_prediction_roundtrip_seconds': durations,
                  'normal_requests_pass_explicit_noise': False,
                  'prediction_timeout_increased': False,
                  'scope': 'saved observation, actual selected model, training-host loopback; laboratory tunnel not tested',
                  'robot_commands_sent': 0}
        (root/'reports/startup_real_model_check.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
