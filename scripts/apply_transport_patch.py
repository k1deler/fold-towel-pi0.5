"""Apply the compact-image client patch to an existing independent bundle. No devices."""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(data):
    return hashlib.sha256(data).hexdigest()


def apply(root, payload):
    root = Path(root).resolve()
    # Never patch a client while its robot runner is active, even if paused.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc/'cmdline').read_bytes().split(b'\0')
            cwd = (proc/'cwd').resolve()
            if any(args[i:i+2] == [b'-m', b'yam_infer.run'] for i in range(len(args)-1)):
                if cwd == root/'inference' or cwd.is_relative_to(root):
                    raise RuntimeError('Robot runner still active; exit it before patching')
        except (OSError, PermissionError):
            continue
    manifest_path = root/'BUNDLE_SHA256.json'
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    for name, expected in manifest.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or digest(path.read_bytes()) != expected:
            raise ValueError('Existing bundle has unreviewed changes: '+name)
    selected = json.loads((root/'deploy_metadata.json').read_text())['model_id']
    if selected != payload['model_id']:
        raise ValueError('Patch is for a different selected checkpoint')
    originals, replacements = {}, {}
    for name, entry in payload['files'].items():
        if name not in ('inference/deploy_pi05/common.py', 'inference/deploy_pi05/client.py',
                        'inference/deploy_pi05/gate.py'):
            raise ValueError('Unexpected patch path')
        path = root/name
        originals[name] = path.read_bytes()
        replacement = base64.b64decode(entry['base64'], validate=True)
        allowed = [entry['before_sha256'], entry['after_sha256']]+entry.get('compatible_before_sha256', [])
        if digest(originals[name]) not in allowed:
            raise ValueError('Unsupported client version: '+name)
        if digest(replacement) != entry['after_sha256']:
            raise ValueError('Patch contents corrupted')
        compile(replacement, name, 'exec')
        replacements[name] = replacement
    backup = root/'patch_backups'/('compact_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True)
    (backup/'BUNDLE_SHA256.json').write_bytes(original_manifest)
    marker = root/'model_gate_passed.json'
    old_marker = marker.read_bytes() if marker.exists() else None
    if old_marker is not None:
        (backup/'model_gate_passed.json').write_bytes(old_marker)
    for name, data in originals.items():
        dest = backup/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    try:
        if marker.exists():
            marker.unlink()
        for name, data in replacements.items():
            (root/name).write_bytes(data)
            manifest[name] = digest(data)
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        test = backup/'test_compact_transport.py'
        test.write_bytes(base64.b64decode(payload['test_base64'], validate=True))
        env = dict(os.environ, PYTHONPATH=str(root/'inference'), PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([str(root/'inference/.venv/bin/python'), '-m', 'pytest', '-q',
                                 '-p', 'no:cacheprovider', str(test)], cwd=root/'inference',
                                env=env, capture_output=True, text=True, timeout=90)
        print(result.stdout, result.stderr)
        if result.returncode:
            raise RuntimeError('Client regression tests failed')
        report = {'status': 'compact_client_patch_installed', 'test_output': result.stdout,
                  'model_id': selected, 'backup': str(backup),
                  'replay_validation_required': True, 'robot_commands_sent': 0}
        (root/'logs').mkdir(exist_ok=True)
        (root/'logs/compact_transport_patch.json').write_text(json.dumps(report, indent=2)+'\n')
    except BaseException:
        for name, data in originals.items():
            (root/name).write_bytes(data)
        manifest_path.write_bytes(original_manifest)
        if old_marker is not None:
            marker.write_bytes(old_marker)
        raise
    print('PI05_COMPACT_TRANSPORT_PATCHED; run validate.sh before any robot start')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--payload', type=Path, required=True)
    a = p.parse_args()
    apply(a.bundle, json.loads(a.payload.read_text()))


if __name__ == '__main__':
    main()
