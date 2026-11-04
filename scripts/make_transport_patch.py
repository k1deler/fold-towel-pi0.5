"""Create a self-contained, hash-checked patch for a previously shipped bundle."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous-payload', type=Path)
    a = p.parse_args()
    manifest = json.loads((a.reference_bundle/'BUNDLE_SHA256.json').read_text())
    metadata = json.loads((a.reference_bundle/'deploy_metadata.json').read_text())
    payload = {'model_id': metadata['model_id'], 'files': {},
               'test_base64': base64.b64encode((ROOT/'tests/test_deployment.py').read_bytes()).decode()}
    previous = json.loads(a.previous_payload.read_text()) if a.previous_payload else None
    for name in ('common.py', 'client.py', 'gate.py'):
        path = 'inference/deploy_pi05/'+name
        data = (ROOT/'deploy_pi05'/name).read_bytes()
        payload['files'][path] = {'before_sha256': manifest[path],
                                 'after_sha256': hashlib.sha256(data).hexdigest(),
                                 'base64': base64.b64encode(data).decode()}
        if previous:
            payload['files'][path]['compatible_before_sha256'] = [previous['files'][path]['after_sha256']]
    patcher = (ROOT/'scripts/apply_transport_patch.py').read_text(encoding='utf-8')
    source = patcher[:patcher.index("if __name__ == '__main__':")]
    source += "\napply(Path.home()/'kideler_deploy/"+a.reference_bundle.name+"', json.loads("+repr(json.dumps(payload))+"))\n"
    compile(source, 'embedded_transport_patch.py', 'exec')
    a.output.write_bytes(('#!/usr/bin/env bash\nset -euo pipefail\npython3 - <<\'PY\'\n'+source+'\nPY\n').encode())
    print('COMPACT_TRANSPORT_PATCH_GENERATED', a.output)


if __name__ == '__main__':
    main()
