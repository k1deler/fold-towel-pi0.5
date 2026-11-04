"""Validate a saved observation against the live inference service, without hardware."""
import argparse
import json
from pathlib import Path
import socket
import time
import uuid

import numpy as np

from .client import RemotePolicy
from .common import (CAMERAS, sha, prepare_frames, validate_info, validate_state,
                     physical_actions)
from .wire import receive, send


def numerical_replay(address, model_id, frames, state, noise):
    """Offline reference comparison only; returned arrays never reach a runner.

    Lossless recorded PNGs are intentionally larger than real-time JPEG inputs.
    Their transfer has its own 10s limit; the separate NORMAL client checks below
    retain the 1s action deadline and are mandatory before granting a gate pass.
    """
    host, port = address.rsplit(':', 1)
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('Use loopback or SSH forwarding')
    _, blobs = prepare_frames(frames, False)
    request_id = uuid.uuid4().hex
    state = validate_state(state)
    noise = np.asarray(noise, dtype=np.float32)
    if noise.shape != (50,32) or not np.isfinite(noise).all():
        raise ValueError('Invalid fixed replay noise')
    with socket.create_connection((host, int(port)), timeout=10) as sock:
        send(sock, {'cmd':'info'})
        info, extra = receive(sock)
        if extra:
            raise ValueError('Unexpected info payload')
        validate_info(info, model_id)
        start = time.monotonic()
        send(sock, {'cmd':'predict', 'task':'fold_towel', 'model_id':model_id,
                    'request_id':request_id, 'qpos':state.tolist(), 'noise':noise.tolist()}, blobs)
        reply, extra = receive(sock)
        elapsed = time.monotonic()-start
    if 'error' in reply:
        raise RuntimeError(reply['error'])
    if extra or reply.get('request_id') != request_id or reply.get('model_id') != model_id:
        raise ValueError('Numerical replay response identity mismatch')
    if elapsed >= 10:
        raise TimeoutError('Offline numerical replay exceeded 10 seconds')
    raw = np.asarray(reply['raw_actions'], dtype=np.float32)
    actions, _ = physical_actions(raw)
    if not np.array_equal(actions, np.asarray(reply['actions'], dtype=np.float32)):
        raise ValueError('Numerical replay gripper protection mismatch')
    return actions, raw, elapsed


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
    action, raw, numerical_seconds = numerical_replay(
        a.address, metadata['model_id'], frames, observation['qpos'], observation['noise'])
    error = float(np.max(np.abs(action-observation['expected_actions'])))
    raw_error = float(np.max(np.abs(raw-observation['expected_raw'])))
    if error > .002 or raw_error > .002:
        raise ValueError('Saved observation / service prediction mismatch')
    # The original fixed-noise PNG replay alone did not exercise either the
    # production JPEG transport or OpenPI's noise=None request signature.
    live_client = RemotePolicy(a.address, jpeg_roundtrip=True,
                               expected_model_id=metadata['model_id'])
    live_durations = []
    live_payloads = []
    try:
        for _ in range(5):
            live_client.predict(frames, observation['qpos'], 'fold_towel')
            live_durations.append(live_client.last_request_seconds)
            live_payloads.append(live_client.last_payload_bytes)
    finally:
        live_client.close()
    result = {'status': 'passed', 'model_id': metadata['model_id'], 'address': a.address,
              'bundle_manifest_sha256': sha(root/'BUNDLE_SHA256.json'),
              'max_abs_error': error, 'raw_max_abs_error': raw_error,
              'numerical_replay_seconds': numerical_seconds, 'created_wall_ns': time.time_ns(),
              'normal_jpeg_roundtrip_seconds': live_durations,
              'normal_jpeg_payload_bytes': live_payloads,
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
