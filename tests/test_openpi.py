"""Uses the installed pinned OpenPI CPU environment; not a model-gradient test."""
import numpy as np
import pytest
pytest.importorskip('openpi')
from openpi import transforms
from fold_pi05.dataset import FoldInputs,FoldOutputs


def test_matches_official_joint_mask_gripper_exclusion():
    q=np.arange(14,dtype=np.float32)/10
    a=np.arange(50*14,dtype=np.float32).reshape(50,14)/100
    im=np.zeros((8,12,3),np.uint8)
    own=FoldInputs()({'state':q,'actions':a,'images':{'left':im,'right':im},'prompt':'Fold the towel.'})
    official=transforms.DeltaActions(transforms.make_bool_mask(6,-1,6,-1))({'state':q.copy(),'actions':a.copy()})
    np.testing.assert_array_equal(own['actions'],official['actions'])
    restored=FoldOutputs()({'state':q,'actions':own['actions']})['actions']
    reference=transforms.AbsoluteActions(transforms.make_bool_mask(6,-1,6,-1))(official)['actions']
    np.testing.assert_array_equal(restored,reference)
