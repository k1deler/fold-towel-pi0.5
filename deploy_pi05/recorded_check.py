"""Recorded-input, real-network replay through the deployment client and server."""
import argparse
import json
from pathlib import Path
import socket
import threading
import time

import numpy as np

from .client import RemotePolicy
from .common import CAMERAS, prepare_frames
from .service import Engine, handle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    a = p.parse_args()
    from fold_pi05.dataset import FoldDataset
    root = a.workspace.resolve()
    ds = FoldDataset(a.data, 'val', 50, root/'reports/split.json')
    sample = ds[0]
    frames = {'right_wrist': sample['images']['right'], 'left_wrist': sample['images']['left']}
    noise = np.random.default_rng(123).normal(size=(50, 32)).astype(np.float32)
    engine = Engine(root)
    # Same recorded RGB values and fixed sampling noise in direct and wire paths.
    engine.infer(frames, sample['state'], noise)
    raw, expected, excursions = engine.infer(frames, sample['state'], noise)
    with socket.create_server(('127.0.0.1', 0)) as server:
        port = server.getsockname()[1]
        def serve_once():
            connection, _ = server.accept()
            handle(engine, connection)
        worker = threading.Thread(target=serve_once, daemon=True)
        worker.start()
        client = RemotePolicy(f'127.0.0.1:{port}', jpeg_roundtrip=False,
                              expected_model_id=engine.model_id)
        try:
            action, images = client.predict(frames, sample['state'], 'fold_towel', noise=noise)
            error = float(np.max(np.abs(action-expected)))
            raw_error = float(np.max(np.abs(client.last_raw-raw)))
            if error > 2e-5 or raw_error > 2e-5:
                raise ValueError('Direct/service replay mismatch')
            if not all(np.array_equal(images[i], frames[name]) for i, name in enumerate(CAMERAS)):
                raise ValueError('Transport altered recorded RGB input')
            if not np.array_equal(action[:, [6, 13]], np.clip(raw[:, [6, 13]], 0, 1)):
                raise ValueError('Absolute gripper clamp mismatch')
            if not np.array_equal(action[:, [0,1,2,3,4,5,7,8,9,10,11,12]],
                                  raw[:, [0,1,2,3,4,5,7,8,9,10,11,12]]):
                raise ValueError('Physical joints were re-anchored after restoration')
            wire_seconds = client.last_request_seconds
        finally:
            client.close()
        worker.join(timeout=3)
        if worker.is_alive():
            raise RuntimeError('Inference connection did not cleanly close')
    # Exercise the real-camera JPEG-q90 preprocessing path with the same saved RGB.
    processed, _ = prepare_frames(frames, True)
    converted = {name: processed[i] for i, name in enumerate(CAMERAS)}
    engine.infer(converted, sample['state'], noise)
    replay = root/'replay'
    replay.mkdir(exist_ok=True)
    np.savez_compressed(replay/'observation.npz', qpos=sample['state'], noise=noise,
                        expected_raw=raw, expected_actions=expected, **frames)
    result = {'status': 'recorded_network_replay_passed', 'model_id': engine.model_id,
              'sample_split': 'val', 'sample_index': 0, 'shape': [50,14],
              'direct_service_max_abs_error': error, 'raw_max_abs_error': raw_error,
              'lossless_pixels_equal': True, 'jpeg_quality_90_path_checked': True,
              'gripper_clipped_values': excursions, 'joint_anchor_added_twice': False,
              'wire_seconds': wire_seconds, 'recorded_at_wall_ns': time.time_ns(),
              'robot_commands_sent': 0, 'robot_tested': False}
    (root/'reports/deployment_recorded_replay.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
