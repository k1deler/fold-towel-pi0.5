import socket
import threading
import time

import pytest

from deploy_pi05.service import handle
from deploy_pi05.wire import receive, send


class InfoEngine:
    def info(self):
        return {'status': 'ok'}


def test_operator_wait_longer_than_old_ten_second_timeout():
    client, server = socket.socketpair()
    client.settimeout(2)
    worker = threading.Thread(target=handle, args=(InfoEngine(), server), daemon=True)
    worker.start()
    try:
        send(client, {'cmd': 'info'})
        assert receive(client) == ({'status': 'ok'}, [])
        time.sleep(11)
        send(client, {'cmd': 'info'})
        assert receive(client) == ({'status': 'ok'}, [])
    finally:
        client.close()
        worker.join(timeout=2)
    assert not worker.is_alive()


def test_incomplete_frame_still_times_out():
    client, server = socket.socketpair()
    client.settimeout(2)
    worker = threading.Thread(target=handle, args=(InfoEngine(), server),
                              kwargs={'frame_timeout': .1}, daemon=True)
    worker.start()
    try:
        client.sendall(b'\x00')
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert client.recv(1) == b''
    finally:
        client.close()


def test_idle_peer_disconnect_exits_handler():
    client, server = socket.socketpair()
    worker = threading.Thread(target=handle, args=(InfoEngine(), server), daemon=True)
    worker.start()
    client.close()
    worker.join(timeout=2)
    assert not worker.is_alive()
