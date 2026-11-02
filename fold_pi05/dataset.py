"""Read verified ACT converted data without copying or changing physical labels."""
from bisect import bisect_right
import json
from pathlib import Path
import numpy as np

JOINTS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPERS = [6, 13]


def split_names(episodes, ratio=.8, seed=1):
    names = sorted(e['name'] for e in episodes)
    if len(names) != len(set(names)) or len(names) < 5:
        raise ValueError('Insufficient or duplicate episodes')
    order = np.random.RandomState(seed).permutation(len(names))
    cut = int(len(names) * ratio)
    if not 0 < cut < len(names):
        raise ValueError('Empty train/validation split')
    return {'train': [names[i] for i in order[:cut]], 'val': [names[i] for i in order[cut:]]}


class FoldDataset:
    def __init__(self, data_root, split, horizon=50, split_path=None, prompt='Fold the towel.'):
        self.root = Path(data_root)
        self.meta = json.loads((self.root/'meta.json').read_text(encoding='utf-8'))
        if (self.meta.get('format'), self.meta.get('fps'), self.meta.get('qpos'), self.meta.get('action')) != (
                'yam_train/jpeg-v1', 30, 'follower', 'leader'):
            raise ValueError('Expected ACT converted leader-target / follower-observation dataset')
        if self.meta.get('image_size') != [480, 640]:
            raise ValueError('Expected original 480x640 wrist images')
        if self.meta.get('state_layout') != 'left_joint0..5,left_gripper,right_joint0..5,right_gripper':
            raise ValueError('Unknown physical state/action layout')
        all_episodes = sorted([e for e in self.meta['episodes'] if e['task'] == 'fold_towel'], key=lambda e:e['name'])
        selected = (json.loads(Path(split_path).read_text()) if split_path else split_names(all_episodes))
        if set(selected['train']) & set(selected['val']):
            raise ValueError('Episode split leakage')
        if set(selected['train']) | set(selected['val']) != {e['name'] for e in all_episodes}:
            raise ValueError('Split must cover exactly the verified fold_towel episodes')
        if split not in ('train', 'val') or horizon < 1:
            raise ValueError('Invalid split or horizon')
        self.episodes = [e for e in all_episodes if e['name'] in set(selected[split])]
        self.horizon, self.prompt = int(horizon), prompt
        counts = [e['length'] - self.horizon + 1 for e in self.episodes]
        if not counts or min(counts) < 1:
            raise ValueError('Every episode must contain a full action horizon; padding is not used')
        self.ends = np.cumsum(counts).tolist()
        self._arrays = None
        self._images = {}

    def __len__(self):
        return self.ends[-1]

    def locate(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = bisect_right(self.ends, index)
        local = index - (self.ends[ep-1] if ep else 0)
        return self.episodes[ep]['start'] + local

    def arrays(self):
        if self._arrays is None:
            self._arrays = {k:np.load(self.root/f'{k}.npy', mmap_mode='r') for k in ('qpos','action')}
        return self._arrays

    def numeric(self, index):
        row = self.locate(index)
        arrays = self.arrays()
        state = np.array(arrays['qpos'][row], dtype=np.float32, copy=True)
        actions = np.array(arrays['action'][row:row+self.horizon], dtype=np.float32, copy=True)
        if state.shape != (14,) or actions.shape != (self.horizon,14) or not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError('Invalid physical state/leader target')
        if np.any(state[GRIPPERS]<0) or np.any(state[GRIPPERS]>1) or np.any(actions[:,GRIPPERS]<0) or np.any(actions[:,GRIPPERS]>1):
            raise ValueError('Recorded physical gripper must be in [0,1]')
        return state, actions

    def image(self, camera, row):
        import cv2
        if camera not in self._images:
            self._images[camera] = (np.load(self.root/f'{camera}.idx.npy',mmap_mode='r'),
                                   np.memmap(self.root/f'{camera}.jpg.bin',dtype=np.uint8,mode='r'))
        idx, raw = self._images[camera]
        off, n = idx[row]
        image = cv2.imdecode(np.asarray(raw[int(off):int(off+n)]),cv2.IMREAD_COLOR)
        if image is None or image.shape != (480,640,3):
            raise ValueError('Cannot decode original wrist JPEG')
        return np.ascontiguousarray(image[:,:,::-1])

    def __getitem__(self, index):
        state, actions = self.numeric(index)
        row = self.locate(index)
        return {'state':state, 'actions':actions, 'prompt':self.prompt,
                'images':{side:self.image(side+'_wrist',row) for side in ('left','right')}}

    def __getstate__(self):
        values = self.__dict__.copy()
        values['_arrays'], values['_images'] = None, {}
        return values


class FoldInputs:
    """Native YAM values; standard joint-delta / absolute-gripper convention."""
    def __call__(self, data):
        state = np.asarray(data['state'],np.float32)
        left, right = (np.asarray(data['images'][s]) for s in ('left','right'))
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('Expected finite 14D follower state')
        for image in (left,right):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError('Expected HWC RGB uint8 images')
        result = {'state':state.copy(),'prompt':data['prompt'],
                  'image':{'base_0_rgb':np.zeros_like(left),'left_wrist_0_rgb':left,'right_wrist_0_rgb':right},
                  'image_mask':{'base_0_rgb':np.bool_(False),'left_wrist_0_rgb':np.bool_(True),'right_wrist_0_rgb':np.bool_(True)}}
        if 'actions' in data:
            actions = np.array(data['actions'],dtype=np.float32,copy=True)
            if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
                raise ValueError('Invalid physical action chunk')
            actions[:,JOINTS] -= state[JOINTS]
            result['actions'] = actions
        return result


class FoldOutputs:
    """Called AFTER OpenPI unnormalizes state/actions; no robot command is sent."""
    def __call__(self, data):
        state = np.asarray(data['state'])
        actions = np.array(data['actions'],dtype=np.float32,copy=True)
        if state.ndim != 1 or state.size < 14 or actions.ndim != 2 or actions.shape[1] < 14:
            raise ValueError('Expected unbatched model state/actions')
        actions = actions[:,:14].copy()
        actions[:,JOINTS] += state[JOINTS]
        if not np.isfinite(actions).all():
            raise ValueError('Nonfinite restored actions')
        # Keep regression excursions visible; protection belongs to a later,
        # separately validated deployment adapter. This project has no live runner.
        return {'actions':actions}
