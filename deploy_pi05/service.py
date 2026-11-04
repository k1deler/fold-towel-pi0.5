"""Pi05 loopback inference only. No robot/camera/CAN imports or initialization."""
import argparse
import json
from pathlib import Path
import socket
import select
import threading

import numpy as np

from .common import (CAMERAS, FORMAT, checkpoint_identity, decode_frames,
                     physical_actions, validate_info, validate_state, verify_checkpoint)
from .wire import receive, send


class Engine:
    def __init__(self, workspace, checkpoint=None):
        from fold_pi05.config import create
        from openpi.policies.policy_config import create_trained_policy
        self.root = Path(workspace).resolve()
        manifest = json.loads((self.root/'reports/selected_checkpoint_manifest.json').read_text())
        checkpoint = Path(checkpoint or manifest['selected_checkpoint']).resolve()
        self.model_id = verify_checkpoint(checkpoint, manifest)
        metadata = json.loads((self.root/'deploy_metadata.json').read_text())
        if metadata['model_id'] != self.model_id:
            raise ValueError('Metadata belongs to a different checkpoint')
        self.metadata = metadata
        cfg = create(self.root/'code/configs/pi05_lora.json', self.root/'vendor/openpi',
                     self.root, checkpoint/'params')
        self.policy = create_trained_policy(cfg, checkpoint)
        self.lock = threading.Lock()

    def info(self):
        return dict(self.metadata['policy_info'])

    def infer(self, frames, qpos, noise=None):
        sample = {'state': validate_state(qpos), 'prompt': 'Fold the towel.',
                  'images': {'left': frames['left_wrist'], 'right': frames['right_wrist']}}
        with self.lock:
            result = self.policy.infer(sample, noise=noise)
        raw = np.asarray(result['actions'], dtype=np.float32)
        actions, excursions = physical_actions(raw)
        return raw, actions, excursions


def dispatch(engine, message, blobs):
    if message.get('cmd') == 'info':
        if blobs:
            raise ValueError('Info request must have no images')
        return engine.info()
    if message.get('cmd') != 'predict' or message.get('task') != 'fold_towel':
        raise ValueError('Unsupported inference request')
    if message.get('model_id') != engine.model_id:
        raise ValueError('Wrong selected-checkpoint identity')
    request_id = message.get('request_id')
    if not isinstance(request_id, str) or len(request_id) != 32:
        raise ValueError('Invalid request identifier')
    noise = message.get('noise')
    if noise is not None:
        noise = np.asarray(noise, dtype=np.float32)
        if noise.shape != (50, 32) or not np.isfinite(noise).all():
            raise ValueError('Invalid deterministic replay noise')
    raw, actions, excursions = engine.infer(decode_frames(blobs),
                                           validate_state(message['qpos']), noise)
    return {'request_id': request_id, 'model_id': engine.model_id,
            'raw_actions': raw.tolist(), 'actions': actions.tolist(),
            'gripper_clipped_values': excursions}


def handle(engine, connection, *, frame_timeout=10):
    connection.settimeout(frame_timeout)
    with connection:
        try:
            while True:
                # Operator alignment, Enter waits and pauses are not a frame transfer.
                # Start the bounded frame read only when bytes become available.
                if not select.select([connection], [], [], 1.0)[0]:
                    continue
                message, blobs = receive(connection)
                try:
                    reply = dispatch(engine, message, blobs)
                    send(connection, reply)
                except Exception as exc:
                    send(connection, {'error': f'{type(exc).__name__}: {exc}'})
                    return
        except (OSError, ValueError, ConnectionError):
            return


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--port', type=int, default=8778)
    a = p.parse_args()
    import jax
    if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
        raise RuntimeError('Select one CUDA GPU explicitly')
    engine = Engine(a.workspace)
    # Compile once using a recorded observation before READY, never a live camera.
    sample = np.load(a.workspace/'replay/observation.npz', allow_pickle=False)
    frames = {name: sample[name] for name in CAMERAS}
    engine.infer(frames, sample['qpos'], sample['noise'])
    engine.infer(frames, sample['qpos'], sample['noise'])
    with socket.create_server(('127.0.0.1', a.port)) as server:
        print('FOLD_PI05_SERVER_READY 127.0.0.1:'+str(a.port), flush=True)
        while True:
            connection, _ = server.accept()
            # One local operator/validation client at a time, one model context.
            handle(engine, connection)


if __name__ == '__main__':
    main()
