"""Validate a saved observation against the live inference service, without hardware."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .client import RemotePolicy
from .common import CAMERAS, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--address', default='127.0.0.1:8778')
    a = p.parse_args()
    root = a.bundle.resolve()
    marker = root/'model_gate_passed.json'
    if marker.exists():
        marker.unlink()  # a failed new validation must not leave a stale pass
    files = json.loads((root/'BUNDLE_SHA256.json').read_text())
    for name, digest in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError('Bundle file differs: '+name)
    metadata = json.loads((root/'deploy_metadata.json').read_text())
    observation = np.load(root/'replay/observation.npz', allow_pickle=False)
    frames = {name: observation[name] for name in CAMERAS}
    client = RemotePolicy(a.address, jpeg_roundtrip=False,
                          expected_model_id=metadata['model_id'])
    try:
        action, _ = client.predict(frames, observation['qpos'], 'fold_towel', noise=observation['noise'])
        error = float(np.max(np.abs(action-observation['expected_actions'])))
        raw_error = float(np.max(np.abs(client.last_raw-observation['expected_raw'])))
        if error > .002 or raw_error > .002:
            raise ValueError('Saved observation / service prediction mismatch')
        # Three steady-state roundtrips detect a path too slow for H50/30Hz.
        durations = []
        for _ in range(3):
            client.predict(frames, observation['qpos'], 'fold_towel', noise=observation['noise'])
            durations.append(client.last_request_seconds)
    finally:
        client.close()
    result = {'status': 'passed', 'model_id': metadata['model_id'], 'address': a.address,
              'bundle_manifest_sha256': sha(root/'BUNDLE_SHA256.json'),
              'max_abs_error': error, 'raw_max_abs_error': raw_error,
              'roundtrip_seconds': durations, 'created_wall_ns': time.time_ns(),
              'robot_commands_sent': 0, 'robot_tested': False}
    marker.write_text(json.dumps(result, indent=2)+'\n')
    print('FOLD_PI05_INSTALL_AND_OFFLINE_REPLAY_PASSED; no hardware started', flush=True)
    print(json.dumps(result, indent=2), flush=True)


def require(bundle):
    root = Path(bundle).resolve()
    marker = json.loads((root/'model_gate_passed.json').read_text())
    metadata = json.loads((root/'deploy_metadata.json').read_text())
    if marker.get('status') != 'passed' or marker['model_id'] != metadata['model_id']:
        raise ValueError('Recorded model replay required')
    if marker['bundle_manifest_sha256'] != sha(root/'BUNDLE_SHA256.json'):
        raise ValueError('Bundle manifest changed after replay')
    for name, digest in json.loads((root/'BUNDLE_SHA256.json').read_text()).items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError('Bundle file changed after replay: '+name)
    if time.time_ns()-marker['created_wall_ns'] > 24*3600*10**9:
        raise ValueError('Replay older than 24 hours; run validate.sh again')
    return metadata['model_id']


if __name__ == '__main__':
    main()
