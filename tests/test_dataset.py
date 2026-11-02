import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from fold_pi05.dataset import FoldDataset, FoldInputs, FoldOutputs, GRIPPERS, JOINTS, split_names


@pytest.fixture
def converted(tmp_path):
    root=tmp_path/'converted';root.mkdir()
    episodes=[{'name':f'fold_towel/episode_{i:04d}','task':'fold_towel','start':i*60,'length':60} for i in range(6)]
    q=np.zeros((360,14),np.float32);q[:,GRIPPERS]=.7
    a=np.zeros_like(q);a[:,GRIPPERS]=.3
    for i in range(6):
        q[i*60:(i+1)*60,0]=i
        a[i*60:(i+1)*60,0]=i+np.arange(60,dtype=np.float32)/100
    np.save(root/'qpos.npy',q);np.save(root/'action.npy',a)
    metadata={'format':'yam_train/jpeg-v1','fps':30,'image_size':[480,640],
              'qpos':'follower','action':'leader','cameras':['right_wrist','left_wrist'],
              'state_layout':'left_joint0..5,left_gripper,right_joint0..5,right_gripper','episodes':episodes}
    (root/'meta.json').write_text(json.dumps(metadata))
    return root,metadata


def test_full_episode_split_and_windows_never_cross_boundaries(converted):
    root,metadata=converted
    split=split_names(metadata['episodes'])
    assert not set(split['train']) & set(split['val'])
    assert split==split_names(list(reversed(metadata['episodes'])))
    train=FoldDataset(root,'train',50)
    assert len(train)==4*11
    for index in (0,10,11,21,len(train)-1):
        row=train.locate(index)
        assert row//60==(row+49)//60
        state,action=train.numeric(index)
        np.testing.assert_array_equal(action,np.load(root/'action.npy')[row:row+50])
        assert state.shape==(14,)
    for bad in (-1,len(train)):
        with pytest.raises(IndexError):train.numeric(bad)


def test_labels_are_leader_at_t_plus_k_not_follower(converted):
    root,_=converted
    ds=FoldDataset(root,'train',50)
    state,action=ds.numeric(0)
    assert state[6]==np.float32(.7) and action[0,6]==np.float32(.3)
    assert action[1,0]!=state[0]


def test_inputs_outputs_roundtrip_preserves_absolute_grippers_and_camera_identity():
    state=np.arange(14,dtype=np.float32)/10
    actions=np.tile(state,(50,1))+.02
    actions[:,GRIPPERS]=.35
    left=np.zeros((8,12,3),np.uint8);left[:,:,0]=255
    right=np.zeros((8,12,3),np.uint8);right[:,:,2]=255
    original=actions.copy()
    encoded=FoldInputs()({'state':state,'actions':actions,'images':{'left':left,'right':right},'prompt':'Fold the towel.'})
    np.testing.assert_array_equal(actions,original)
    np.testing.assert_allclose(encoded['actions'][:,JOINTS],.02,atol=1e-7)
    np.testing.assert_array_equal(encoded['actions'][:,GRIPPERS],original[:,GRIPPERS])
    assert not encoded['image_mask']['base_0_rgb'] and encoded['image_mask']['left_wrist_0_rgb']
    np.testing.assert_array_equal(encoded['image']['left_wrist_0_rgb'],left)
    np.testing.assert_array_equal(encoded['image']['right_wrist_0_rgb'],right)
    padded=np.pad(encoded['actions'],((0,0),(0,18)))
    decoded=FoldOutputs()({'state':np.pad(state,(0,18)),'actions':padded})['actions']
    np.testing.assert_allclose(decoded,original,atol=1e-7)


@pytest.mark.parametrize('change',[{'fps':20},{'qpos':'leader'},{'action':'future_follower_absolute'},
                                 {'image_size':[240,320]},{'state_layout':'right_then_left'}])
def test_wrong_data_contract_is_rejected(converted,change):
    root,metadata=converted;metadata.update(change)
    (root/'meta.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):FoldDataset(root,'train')


def test_leaking_or_missing_episode_split_is_rejected(converted,tmp_path):
    root,metadata=converted
    split=split_names(metadata['episodes']);split['val'].append(split['train'][0])
    p=tmp_path/'split.json';p.write_text(json.dumps(split))
    with pytest.raises(ValueError,match='leakage'):FoldDataset(root,'train',split_path=p)


def test_numeric_views_are_copies_and_pickling_drops_memmaps(converted):
    root,_=converted;ds=FoldDataset(root,'train');state,actions=ds.numeric(0)
    state[0]=99;actions[0,0]=99
    assert ds.numeric(0)[0][0]!=99 and ds.numeric(0)[1][0,0]!=99
    restored=pickle.loads(pickle.dumps(ds))
    assert restored._arrays is None
    np.testing.assert_array_equal(restored.numeric(0)[1],ds.numeric(0)[1])


def test_outputs_leave_gripper_regression_visible_not_silently_safe():
    values=np.zeros((50,32),np.float32);values[:,GRIPPERS]=1.2
    result=FoldOutputs()({'state':np.zeros(32),'actions':values})['actions']
    assert np.all(result[:,GRIPPERS]>1)
    values[0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):FoldOutputs()({'state':np.zeros(32),'actions':values})


def test_action_horizon_requires_real_targets_no_padding(converted):
    root,_=converted
    with pytest.raises(ValueError,match='full action horizon'):FoldDataset(root,'val',100)
