import importlib.util
from pathlib import Path

path=Path(__file__).resolve().parents[1]/'scripts/train_pipeline.py'
spec=importlib.util.spec_from_file_location('fold_pi05_train_pipeline',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_only_idle_sufficient_memory_cards_are_eligible():
    rows=[dict(index=0,used_mib=72038,total_mib=97871,utilization=100),
          dict(index=1,used_mib=0,total_mib=97871,utilization=0),
          dict(index=2,used_mib=500,total_mib=97871,utilization=95),
          dict(index=3,used_mib=0,total_mib=16000,utilization=0)]
    assert module.idle_candidates(rows)==[1]


def test_no_busy_card_is_treated_as_free_from_low_utilization_only():
    assert module.idle_candidates([dict(index=0,used_mib=29617,total_mib=97871,utilization=0)])==[]
