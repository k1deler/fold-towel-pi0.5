import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('fold_eval_pipeline', scripts/'evaluate_pipeline.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def result(checkpoint, error=.1):
    return dict(checkpoint=str(checkpoint), samples=2, indices=[7, 42], joint_mae_rad=error,
                joint_rmse_rad=error, gripper_mae=.1, first_step_joint_mae_rad=error,
                hold_joint_mae_rad=.2, raw_gripper_out_of_range_fraction=.1,
                median_inference_seconds=.05, robot_commands_sent=0,
                predictions_clipped_before_metrics=False)


def test_invalid_or_clipped_predictions_cannot_be_accepted(tmp_path):
    value = result(tmp_path)
    queue.validate_result(value, tmp_path, 2)
    for key, bad in [('joint_mae_rad', float('nan')), ('raw_gripper_out_of_range_fraction', 1.1),
                     ('predictions_clipped_before_metrics', True), ('robot_commands_sent', 1)]:
        invalid = dict(value, **{key: bad})
        with pytest.raises(ValueError):
            queue.validate_result(invalid, tmp_path, 2)


def test_unpaired_windows_are_rejected(tmp_path):
    a, b = result(tmp_path/'a'), result(tmp_path/'b')
    b['indices'] = [1, 2]
    with pytest.raises(ValueError):
        queue.comparison([a, b])


def test_selection_keeps_excursions_visible_and_never_deploys(tmp_path):
    summary = queue.comparison([result(tmp_path/'a', .3), result(tmp_path/'b', .1)])
    assert summary['selected_checkpoint'] == str(tmp_path/'b')
    assert summary['beats_hold_baseline']
    assert summary['raw_gripper_excursions_require_deployment_handling']
    assert not summary['automatic_deployment']


def test_duplicate_or_wrong_checkpoint_report_is_rejected(tmp_path):
    value = result(tmp_path)
    value['indices'] = [7, 7]
    with pytest.raises(ValueError):
        queue.validate_result(value, tmp_path, 2)
    with pytest.raises(ValueError):
        queue.validate_result(result(tmp_path/'other'), tmp_path, 2)


def test_explicit_sharing_uses_free_memory_and_prefers_lower_load():
    rows = [dict(index=0, used_mib=29617, total_mib=97871, utilization=0),
            dict(index=1, used_mib=49901, total_mib=97871, utilization=100),
            dict(index=2, used_mib=60000, total_mib=97871, utilization=0)]
    assert queue.eligible_gpus(rows) == []
    assert queue.eligible_gpus(rows, True) == [0, 1]


def test_reserve_is_kept_above_allocator_budget():
    rows = [dict(index=0, used_mib=60000, total_mib=97871, utilization=0)]
    assert queue.eligible_gpus(rows, True, 32768, 8192) == []
    assert queue.eligible_gpus(rows, True, 16384, 8192) == [0]
