import json
from pathlib import Path
import pytest
from fold_pi05.config import read_settings

ROOT=Path(__file__).resolve().parents[1]


def test_initial_config_contract():
    s=read_settings(ROOT/'configs/pi05_lora.json')
    assert s['model']=='pi05' and s['method']=='lora' and s['action_horizon']==50
    assert s['gripper_action_mode']=='absolute_0_1' and not s['wandb_enabled']


@pytest.mark.parametrize('change',[{'action_offset':1},{'task':'put_block_in_bowl'},
    {'method':'full'},{'model':'pi0'},{'action_dim':14},{'physical_action_dim':32},
    {'fsdp_devices':4},{'warmup_steps':10000},{'action_horizon':0},{'batch_size':0}])
def test_invalid_contract_rejected(tmp_path,change):
    s=json.loads((ROOT/'configs/pi05_lora.json').read_text());s.update(change)
    p=tmp_path/'settings.json';p.write_text(json.dumps(s))
    with pytest.raises(ValueError):read_settings(p)
