#!/usr/bin/env bash
set -euo pipefail
PY="$HOME/output/envs/yam-openpi/bin/python"
"$PY" - <<'PY'
from pathlib import Path
import datetime
import os
import subprocess
import sys

root = Path.home()/'output/personal/k1deler/fold_towel_pi05_20261102'
code = root/'code'
service = code/'deploy_pi05/service.py'
original = service.read_text(encoding='utf-8')
old = 'def handle(engine, connection):\n    connection.settimeout(10)'
new = 'def handle(engine, connection, *, frame_timeout=10):\n    connection.settimeout(frame_timeout)'
read_old = '            while True:\n                message, blobs = receive(connection)'
read_new = '            while True:\n                if not select.select([connection], [], [], 1.0)[0]:\n                    continue\n                message, blobs = receive(connection)'
if 'select.select([connection]' not in original:
    if original.count(old) != 1 or original.count(read_old) != 1:
        raise SystemExit('Unexpected service version; no changes made')
    updated = original.replace('import socket\n', 'import socket\nimport select\n').replace(old, new).replace(read_old, read_new)
else:
    updated = original
compile(updated, str(service), 'exec')
backup = service.with_name('service.py.before_idle_fix_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.bak')
backup.write_text(original, encoding='utf-8')
service.write_text(updated, encoding='utf-8')
(code/'tests/test_service_idle.py').write_text("import socket\nimport threading\nimport time\n\nimport pytest\n\nfrom deploy_pi05.service import handle\nfrom deploy_pi05.wire import receive, send\n\n\nclass InfoEngine:\n    def info(self):\n        return {'status': 'ok'}\n\n\ndef test_operator_wait_longer_than_old_ten_second_timeout():\n    client, server = socket.socketpair()\n    client.settimeout(2)\n    worker = threading.Thread(target=handle, args=(InfoEngine(), server), daemon=True)\n    worker.start()\n    try:\n        send(client, {'cmd': 'info'})\n        assert receive(client) == ({'status': 'ok'}, [])\n        time.sleep(11)\n        send(client, {'cmd': 'info'})\n        assert receive(client) == ({'status': 'ok'}, [])\n    finally:\n        client.close()\n        worker.join(timeout=2)\n    assert not worker.is_alive()\n\n\ndef test_incomplete_frame_still_times_out():\n    client, server = socket.socketpair()\n    client.settimeout(2)\n    worker = threading.Thread(target=handle, args=(InfoEngine(), server),\n                              kwargs={'frame_timeout': .1}, daemon=True)\n    worker.start()\n    try:\n        client.sendall(b'\\x00')\n        worker.join(timeout=2)\n        assert not worker.is_alive()\n        assert client.recv(1) == b''\n    finally:\n        client.close()\n\n\ndef test_idle_peer_disconnect_exits_handler():\n    client, server = socket.socketpair()\n    worker = threading.Thread(target=handle, args=(InfoEngine(), server), daemon=True)\n    worker.start()\n    client.close()\n    worker.join(timeout=2)\n    assert not worker.is_alive()\n", encoding='utf-8')
env = dict(os.environ, PYTHONPATH=str(code), PYTHONDONTWRITEBYTECODE='1', JAX_PLATFORMS='cpu')
try:
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_service_idle.py', 'tests/test_deployment.py'], cwd=code, env=env, timeout=90)
    if result.returncode:
        raise RuntimeError('Regression tests failed')
except BaseException:
    service.write_text(original, encoding='utf-8')
    raise
print('PI05_IDLE_WAIT_FIXED_AND_TESTED; restart inference service explicitly; no robot commands')
print('Backup:', backup)

PY
