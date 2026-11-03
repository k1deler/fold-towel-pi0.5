"""Build a separate YAM operator bundle. Does not open hardware or alter ACT."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def replace(path, old, new):
    text = path.read_text(encoding='utf-8')
    if text.count(old) != 1:
        raise ValueError('Unsupported original runtime: '+str(path))
    path.write_text(text.replace(old, new), encoding='utf-8')


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--act-project', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True, help='Recorded replay and verified model metadata')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    destination = a.output.resolve()
    if destination.exists():
        raise FileExistsError('Preserve existing bundle; choose a new output')
    source = (a.act_project/'inference').resolve()
    if destination.is_relative_to(source):
        raise ValueError('Output must be independent of ACT')
    destination.mkdir(parents=True)
    shutil.copy2(ROOT/'THIRD_PARTY.md', destination/'THIRD_PARTY.md')
    # Preserve upstream attribution in the standalone delivery as well as Git.
    for name in ('THIRD_PARTY.md', 'LICENSE', 'NOTICE'):
        if (a.act_project/name).is_file():
            shutil.copy2(a.act_project/name, destination/('ACT_'+name))
    target = destination/'inference'
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        '.git', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache', '*.pt', '*.pth',
        '*.key', '*.pem', 'robot.env', 'logs', 'runs_infer'))
    shutil.copytree(ROOT/'deploy_pi05', target/'deploy_pi05', ignore=shutil.ignore_patterns('__pycache__'))
    replace(target/'yam_infer/run.py', 'from yam_infer.policy.server import RemotePolicy',
            'from deploy_pi05.client import RemotePolicy')
    replace(target/'yam_infer/config.py', 'FORMATS = ("act_v2", "act_ref")',
            'FORMATS = ("act_v2", "act_ref", "pi05")')
    replace(target/'yam_infer/config.py', '"act_ref": "yam_train/act_ref-v1"}',
            '"act_ref": "yam_train/act_ref-v1", "pi05": "openpi/pi05-lora"}')
    # The original scheduler holds the final command indefinitely. The copied
    # pi05 runner instead stops this episode when an observation-aligned plan
    # expires. New inference still runs asynchronously at the original cadence.
    replace(target/'yam_infer/run.py', 'scheduler.add(t_img - t_shift, chunk)',
            'if time.monotonic() - t_img >= 49 / policy.fps:\n'
            '                        raise TimeoutError("Prediction expired before scheduling")\n'
            '                    scheduler.add(t_img - t_shift, chunk)\n'
            '                    shared["plan_deadline"] = t_img + 49 / policy.fps')
    replace(target/'yam_infer/run.py', 'raw = None if paused.is_set() else scheduler.target_at(now)',
            'if not paused.is_set() and shared.get("plan_deadline") is not None and now >= shared["plan_deadline"]:\n'
            '                raise TimeoutError("Observation-aligned pi05 plan expired")\n'
            '            raw = None if paused.is_set() else scheduler.target_at(now)')
    replace(target/'yam_infer/run.py', 'scheduler.clear()\n                    paused.clear()',
            'scheduler.clear()\n                    shared.pop("plan_deadline", None)\n                    paused.clear()')
    replace(target/'yam_infer/run.py', 'if cams is not None:\n            cams.close()',
            'if cams is not None:\n            cams.close()\n'
            '        if hasattr(policy, "close"):\n            policy.close()')
    # Keep manual STOP/fault handling: the copied runner does not automatically
    # invoke recover() while parking after a fault.
    replace(target/'yam_infer/run.py', 'print("  worker not in control -> reset fault and re-open a session")\n                    arms.recover()',
            'raise RuntimeError("Worker fault/session lost; manual recovery required")')
    write(target/'configs/models.yaml', 'model:\n  ckpt: models/pi05_selected\n  format: pi05\n'
          '  task: fold_towel\n  action_mode: abs\n  compensate_action_shift: false\n')
    # Two original tests assert the original repository's example task/model.
    # Replace those fixture-specific checks with the actual deployment identity,
    # retaining the remaining model, control, watchdog, recorder and RPC tests.
    test = target/'tests/test_models.py'
    text = test.read_text(encoding='utf-8')
    pattern = r'def test_project_models_config\(\).*?(?=GOOD = )'
    replacement = '''def test_project_models_config() -> None:
    m = load_model_config(DEFAULT_MODELS_CONFIG)
    assert (m.format, m.task, m.action_mode) == ("pi05", "fold_towel", "abs")
    assert m.ckpt.is_absolute() and m.ckpt.name == "pi05_selected"


def test_pi05_identity_contract() -> None:
    m = load_model_config(DEFAULT_MODELS_CONFIG)
    check_model_matches(m, "openpi/pi05-lora", ["fold_towel"], "abs")
    with pytest.raises(ValueError, match="format"):
        check_model_matches(m, "yam_train/act_ref-v1", ["fold_towel"], "abs")


'''
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise ValueError('Unsupported original model test fixtures')
    test.write_text(text, encoding='utf-8')
    replace(target/'scripts/check_env.py', 'm = load_model_config()\n    if not m.ckpt.is_file():',
            'm = load_model_config()\n'
            '    if m.format == "pi05":\n'
            '        import json\n'
            '        from deploy_pi05.common import validate_info\n'
            '        metadata = json.loads((ROOT.parent / "deploy_metadata.json").read_text())\n'
            '        validate_info(metadata["policy_info"], metadata["model_id"])\n'
            '        check_model_matches(m, "openpi/pi05-lora", ["fold_towel"], "abs")\n'
            '        return "pi05 metadata verified; service replay is a separate step"\n'
            '    if not m.ckpt.is_file():')
    for name in ('deploy_metadata.json',):
        shutil.copy2(a.assets/name, destination/name)
    shutil.copytree(a.assets/'replay', destination/'replay')
    shutil.copytree(ROOT/'tests', target/'pi05_tests', ignore=shutil.ignore_patterns('__pycache__'))
    # Only deployment tests belong in the robot environment; OpenPI training is
    # intentionally isolated on the inference host.
    for file in (target/'pi05_tests').glob('*.py'):
        if file.name != 'test_deployment.py':
            file.unlink()
    for name in ('deployment_recorded_replay.json', 'selected_checkpoint_manifest.json'):
        shutil.copy2(a.assets/'reports'/name, destination/name)
    write(destination/'install_bundle.sh', '''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/inference"
bash scripts/setup_robot.sh --dev
.venv/bin/python -m pytest -q -p no:cacheprovider tests pi05_tests
echo 'FOLD_PI05_FRONTEND_INSTALLED; no hardware initialized; next run validate.sh after service/tunnels'
''')
    write(destination/'validate.sh', '''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/inference"
exec .venv/bin/python -m deploy_pi05.gate --bundle "$ROOT" --address 127.0.0.1:8778
''')
    write(destination/'scripts/run_robot.sh', '''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/inference"
export PI05_EXPECTED_MODEL_ID
PI05_EXPECTED_MODEL_ID=$(.venv/bin/python -c 'import sys; from deploy_pi05.gate import require; print(require(sys.argv[1]))' "$ROOT")
# Reconnect and validate model identity before free_hw or any real-device code.
.venv/bin/python -c 'from deploy_pi05.client import RemotePolicy; p=RemotePolicy("127.0.0.1:8778"); p.close(); print("PI05_MODEL_GATE_OK")'
exec bash scripts/run_robot.sh --remote 127.0.0.1:8778 "$@"
''')
    files = {}
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            if path.suffix == '.py':
                compile(path.read_text(encoding='utf-8'), str(path), 'exec')
            files[path.relative_to(destination).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(destination/'BUNDLE_SHA256.json', json.dumps(files, indent=2)+'\n')
    print('FOLD_PI05_OPERATOR_BUNDLE_BUILT; no install-specific replay pass created')


if __name__ == '__main__':
    main()
