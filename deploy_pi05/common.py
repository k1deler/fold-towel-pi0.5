"""Physical, image and checkpoint contracts shared by service and robot client."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

CAMERAS = ['right_wrist', 'left_wrist']
FORMAT = 'openpi/pi05-lora'
JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIPPERS = [6, 13]
HORIZON = 50
FPS = 30
MAX_REQUEST_SECONDS = 1.0  # under the 49/30 second observation-aligned plan


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint_identity(manifest):
    return hashlib.sha256(json.dumps(manifest['files'], sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def verify_checkpoint(checkpoint, manifest):
    checkpoint = Path(checkpoint).resolve()
    for entry in manifest['files']:
        p = (checkpoint/entry['path']).resolve()
        if not p.is_relative_to(checkpoint) or not p.is_file():
            raise ValueError('Invalid/missing checkpoint member: '+entry['path'])
        if p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
            raise ValueError('Checkpoint SHA mismatch: '+entry['path'])
    if not (checkpoint/'params/_METADATA').is_file():
        raise ValueError('Missing real inference parameters')
    return checkpoint_identity(manifest)


def validate_state(state):
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError('Expected finite follower state [14]')
    if np.any(state[GRIPPERS] < 0) or np.any(state[GRIPPERS] > 1):
        raise ValueError('Follower gripper scale must be absolute [0,1]')
    return state


def physical_actions(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (HORIZON, 14) or not np.isfinite(raw).all():
        raise ValueError('Expected finite restored physical actions [50,14]')
    clipped = raw.copy()
    g = raw[:, GRIPPERS]
    excursions = int(((g < 0) | (g > 1)).sum())
    clipped[:, GRIPPERS] = np.clip(g, 0, 1)
    # No follower anchor addition here: OpenPI FoldOutputs already restores
    # the 12 joint deltas; the two grippers are absolute, including in training.
    return clipped, excursions


def prepare_frames(frames, jpeg_roundtrip):
    images, blobs = [], []
    for camera in CAMERAS:
        image = np.asarray(frames[camera])
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError('Expected original HWC RGB uint8 wrist images [480,640,3]')
        if jpeg_roundtrip:
            ok, data = cv2.imencode('.jpg', np.ascontiguousarray(image[:, :, ::-1]),
                                    [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError('Training JPEG preprocessing failed')
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError('Training JPEG decoding failed')
            image = bgr[:, :, ::-1]
            # Send the EXACT bytes from the one JPEG-q90 training preprocessing
            # encode. A PNG of its decoded pixels is much larger over SSH; an
            # additional JPEG encode would change those pixels and is forbidden.
            blob = data.tobytes()
        else:
            # Already-decoded recorded observations need lossless transport.
            ok, encoded = cv2.imencode('.png', np.ascontiguousarray(image[:, :, ::-1]))
            if not ok:
                raise ValueError('Lossless image encoding failed')
            blob = encoded.tobytes()
        images.append(np.ascontiguousarray(image))
        blobs.append(blob)
    return np.stack(images), blobs


def decode_frames(blobs):
    if len(blobs) != 2:
        raise ValueError('Expected two wrist images')
    frames = {}
    for name, data in zip(CAMERAS, blobs):
        if not (data.startswith(b'\x89PNG\r\n\x1a\n') or data.startswith(b'\xff\xd8')):
            raise ValueError('Expected lossless PNG or original preprocessing JPEG')
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (480, 640, 3):
            raise ValueError('Invalid wrist image size')
        frames[name] = np.ascontiguousarray(image[:, :, ::-1])
    return frames


def validate_info(info, expected_model_id):
    expected = dict(format=FORMAT, tasks=['fold_towel'], cameras=CAMERAS,
                    image_size=[480, 640], fps=FPS, chunk_size=HORIZON,
                    action_mode='abs', action_shift=0, model_id=expected_model_id)
    for k, value in expected.items():
        if info.get(k) != value:
            raise ValueError('Wrong service contract: '+k)
    for k in ('action_min', 'action_max'):
        v = np.asarray(info[k])
        if v.shape != (14,) or not np.isfinite(v).all():
            raise ValueError('Invalid training physical bounds')
    if np.any(np.asarray(info['action_min']) > np.asarray(info['action_max'])):
        raise ValueError('Inverted training physical bounds')
    validate_state(info['start_pose']['fold_towel'])
