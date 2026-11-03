import socket
import struct
import threading
import time

import numpy as np
import pytest

from deploy_pi05.common import (CAMERAS, decode_frames, physical_actions, prepare_frames,
                                validate_info, validate_state)
from deploy_pi05.wire import MAX_HEADER, receive, send
from deploy_pi05.client import RemotePolicy


def frames():
    image = np.zeros((480,640,3), np.uint8)
    image[:, :320, 0] = 217
    image[:, 320:, 2] = 93
    return {name: image.copy() for name in CAMERAS}


def test_lossless_transport_and_training_jpeg():
    original = frames()
    images, blobs = prepare_frames(original, False)
    decoded = decode_frames(blobs)
    for i, name in enumerate(CAMERAS):
        np.testing.assert_array_equal(decoded[name], original[name])
        np.testing.assert_array_equal(images[i], original[name])
    images, blobs = prepare_frames(original, True)
    for i, name in enumerate(CAMERAS):
        np.testing.assert_array_equal(decode_frames(blobs)[name], images[i])


def test_only_absolute_grippers_are_clipped_and_joints_are_not_reanchored():
    raw = np.ones((50,14), np.float32)*2
    raw[:,6] = -.4
    raw[:,13] = 1.6
    result, count = physical_actions(raw)
    assert count == 100
    assert np.all(result[:,6] == 0) and np.all(result[:,13] == 1)
    np.testing.assert_array_equal(result[:,0], raw[:,0])
    assert raw[0,6] == np.float32(-.4)


@pytest.mark.parametrize('raw', [np.zeros((100,14)), np.zeros((50,32)), np.full((50,14), np.nan)])
def test_invalid_output_is_rejected(raw):
    with pytest.raises(ValueError):
        physical_actions(raw)


def test_bad_state_and_wire_length_are_rejected():
    with pytest.raises(ValueError):
        validate_state(np.full(14, np.inf))
    a, b = socket.socketpair()
    try:
        a.sendall(struct.pack('!I', MAX_HEADER+1))
        with pytest.raises(ValueError):
            receive(b)
    finally:
        a.close(); b.close()


def info():
    return dict(format='openpi/pi05-lora', tasks=['fold_towel'], cameras=CAMERAS,
                image_size=[480,640], fps=30, chunk_size=50, action_mode='abs',
                action_shift=0, model_id='test', action_min=[-2]*14, action_max=[2]*14,
                start_pose={'fold_towel':[0]*14}, device='fake')


@pytest.mark.parametrize('fault', ['disconnect', 'wrong_id', 'expired'])
def test_network_failures_never_return_actions(fault, monkeypatch):
    with socket.create_server(('127.0.0.1',0)) as server:
        def worker():
            conn, _ = server.accept()
            with conn:
                receive(conn); send(conn, info())
                request, _ = receive(conn)
                if fault == 'disconnect':
                    return
                if fault == 'expired':
                    time.sleep(.1)
                send(conn, dict(request_id='wrong' if fault=='wrong_id' else request['request_id'],
                                model_id='test', raw_actions=np.zeros((50,14)).tolist(),
                                actions=np.zeros((50,14)).tolist()))
        thread = threading.Thread(target=worker, daemon=True); thread.start()
        client = RemotePolicy('127.0.0.1:'+str(server.getsockname()[1]),
                              expected_model_id='test', jpeg_roundtrip=False)
        if fault == 'expired':
            monkeypatch.setattr('deploy_pi05.client.MAX_REQUEST_SECONDS', .05)
        try:
            with pytest.raises((ConnectionError, ValueError, TimeoutError)):
                client.predict(frames(), np.zeros(14), 'fold_towel')
        finally:
            client.close()
        thread.join(timeout=2)


def test_model_identity_is_required_and_verified():
    with pytest.raises(ValueError):
        validate_info(info(), 'different')
