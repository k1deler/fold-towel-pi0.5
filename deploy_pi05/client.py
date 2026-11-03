"""Original YAM Policy interface; lossless transport and bounded absolute grippers."""
import json
import os
import socket
import time
import uuid

import numpy as np

from .common import (CAMERAS, MAX_REQUEST_SECONDS, physical_actions, prepare_frames,
                     validate_info, validate_state)
from .wire import receive, send


class RemotePolicy:
    def __init__(self, address, jpeg_roundtrip=True, timeout=MAX_REQUEST_SECONDS,
                 expected_model_id=None):
        host, port = address.rsplit(':', 1)
        if host not in ('127.0.0.1', 'localhost'):
            raise ValueError('Use loopback or an SSH tunnel to the model service')
        expected_model_id = expected_model_id or os.environ.get('PI05_EXPECTED_MODEL_ID')
        if not expected_model_id:
            raise ValueError('Missing selected-checkpoint identity')
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.jpeg_roundtrip = jpeg_roundtrip
        self.expected_model_id = expected_model_id
        self.last_raw = None
        self.last_gripper_excursions = 0
        self.last_request_seconds = None
        try:
            send(self.sock, {'cmd': 'info'})
            info, blobs = receive(self.sock)
            if blobs:
                raise ValueError('Unexpected info payload')
            validate_info(info, expected_model_id)
            self._info = info
            for key in ('format', 'action_mode', 'action_shift', 'fps', 'chunk_size', 'tasks', 'cameras'):
                setattr(self, key, info[key])
            self.image_size = tuple(info['image_size'])
            self.a_min = np.asarray(info['action_min'], dtype=np.float64)
            self.a_max = np.asarray(info['action_max'], dtype=np.float64)
        except BaseException:
            self.sock.close()
            raise

    def info(self):
        return self._info

    def predict(self, frames, qpos, task, *, noise=None):
        if task != 'fold_towel':
            raise ValueError('This service only supports fold_towel')
        qpos = validate_state(qpos)
        started = time.monotonic()
        images, blobs = prepare_frames(frames, self.jpeg_roundtrip)
        request_id = uuid.uuid4().hex
        message = {'cmd': 'predict', 'qpos': qpos.tolist(), 'task': task,
                   'request_id': request_id, 'model_id': self.expected_model_id}
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)
            if noise.shape != (50, 32) or not np.isfinite(noise).all():
                raise ValueError('Invalid replay noise')
            message['noise'] = noise.tolist()
        send(self.sock, message, blobs)
        reply, extra = receive(self.sock)
        elapsed = time.monotonic()-started
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        if extra or reply.get('request_id') != request_id or reply.get('model_id') != self.expected_model_id:
            raise ValueError('Stale or mismatched inference response')
        if elapsed >= MAX_REQUEST_SECONDS:
            raise TimeoutError('Expired inference response: discard the plan')
        raw = np.asarray(reply['raw_actions'], dtype=np.float32)
        actions, excursions = physical_actions(raw)
        returned = np.asarray(reply['actions'], dtype=np.float32)
        if not np.array_equal(returned, actions):
            raise ValueError('Server/client gripper protection mismatch')
        self.last_raw = raw
        self.last_gripper_excursions = excursions
        self.last_request_seconds = elapsed
        return actions, images

    def close(self):
        self.sock.close()
