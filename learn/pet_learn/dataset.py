"""Training data for the driving model (see MODEL.md): per-step inputs and actions, prediction targets,
and windows of consecutive steps for predicting up to H steps ahead."""
import numpy as np

from pet_learn.episodes import usable

FORWARD_CAP, TURN_CAP = 0.3, 1.5  # the safety layer's caps (m/s, rad/s): actions are scaled by them
SCAN_CAP = 4.0  # lidar sectors are capped at this (m): beyond it, distance matters little for driving
H = 10  # steps ahead to predict (1 s at 10 Hz)
POSE_SCALE = (0.3, 0.3, 1.0)  # pose change targets (m, m, rad) are divided by these: about 1 s of driving

# Prediction targets: name -> (size, binary?)
TARGETS = {
    'scan': (60, False),
    'floor': (1, False),
    'odom': (2, False),
    'wheels': (2, False),
    'current': (2, False),
    'bump': (1, True),
    'slip': (1, True),
}


def step_targets(steps: dict) -> dict:
    """Each step's readings, as targets: name -> (n, size) float32, in the units the inputs use."""
    return {
        'scan': np.nan_to_num(np.clip(steps['scan'], 0, SCAN_CAP), nan=SCAN_CAP) / SCAN_CAP,
        'floor': np.nan_to_num(np.clip(steps['mouse_speed'], 0, 1.0))[:, None] / FORWARD_CAP,
        'odom': steps['odom_vel'] / np.array([FORWARD_CAP, TURN_CAP], np.float32),
        'wheels': steps['wheel_vels'] / 10.0,  # rad/s; 0.3 m/s is about 8 rad/s
        'current': steps['wheel_current'] / 1000.0,  # A
        'bump': steps['bump'].any(axis=1)[:, None].astype(np.float32),
        'slip': steps['slipping'][:, None].astype(np.float32),
    }


def actions(steps: dict) -> np.ndarray:
    return (steps['cmd'] / np.array([FORWARD_CAP, TURN_CAP], np.float32)).astype(np.float32)


def inputs(steps: dict) -> np.ndarray:
    """What the encoder sees at each step: every target reading, plus wheel stall and the previous
    drive command."""
    t = step_targets(steps)
    previous = np.vstack([np.zeros((1, 2), np.float32), actions(steps)[:-1]])
    stall = steps['stall'][:, None].astype(np.float32)
    bump_sides = steps['bump'].astype(np.float32)  # left, centre, right: more than the target's any-bump
    parts = [t['scan'], t['floor'], t['odom'], t['wheels'], t['current'], bump_sides, t['slip'], stall, previous]
    return np.concatenate(parts, axis=1).astype(np.float32)


def pose_changes(pose: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """For windows of step indices idx (B, h+1): how the pose changed from each window's first step,
    in the robot's frame there, scaled. (B, h+1, 3)"""
    p = pose[idx]
    x0, y0, yaw0 = p[:, :1, 0], p[:, :1, 1], p[:, :1, 2]
    c, s = np.cos(yaw0), np.sin(yaw0)
    dx, dy = p[..., 0] - x0, p[..., 1] - y0
    dyaw = np.arctan2(np.sin(p[..., 2] - yaw0), np.cos(p[..., 2] - yaw0))
    return np.stack([(c * dx + s * dy) / POSE_SCALE[0], (-s * dx + c * dy) / POSE_SCALE[1],
                     dyaw / POSE_SCALE[2]], axis=-1).astype(np.float32)


class SessionData:
    """One converted session's model data, with windows split by time into training and held-out."""

    def __init__(self, steps: dict, holdout: float = 0.2, h: int = H):
        self.steps = steps
        self.h = h
        self.inputs = inputs(steps)
        self.actions = actions(steps)
        self.targets = step_targets(steps)
        self.pose = steps['pose'].astype(np.float64)
        ok = usable(steps)
        n = len(ok)
        # A window starting at t needs steps t..t+h all usable.
        bad = np.concatenate([[0], np.cumsum(~ok)])
        starts = np.flatnonzero(bad[np.arange(n - h) + h + 1] - bad[np.arange(n - h)] == 0)
        split = int(n * (1 - holdout))
        self.train = starts[starts + h < split]
        self.val = starts[starts >= split]


def mirror_inputs(x: np.ndarray) -> np.ndarray:
    """Encoder inputs (..., d) flipped left to right (see mirror)."""
    x = x.copy()
    # Input layout (inputs()): scan 60, floor 1, odom 2, wheels 2, current 2, bump sides 3, slip, stall, previous 2
    x[..., 0:60] = x[..., 59::-1]
    x[..., 62] = -x[..., 62]  # turn rate
    x[..., 63:65] = x[..., 64:62:-1]  # wheels left <-> right
    x[..., 65:67] = x[..., 66:64:-1]  # currents
    x[..., 67:70] = x[..., 69:66:-1]  # bump left <-> right
    x[..., 73] = -x[..., 73]  # previous turn command
    return x


def mirror(batch: dict) -> dict:
    """The same windows flipped left to right: physics is symmetric that way, so this doubles the
    data honestly. Works on raw (not yet standardized) batches: reverses the lidar sectors (they're
    symmetric about straight ahead), negates turns, swaps left and right."""
    out = dict(batch)
    out['inputs'] = mirror_inputs(batch['inputs'])
    a = batch['actions'].copy()
    a[..., 1] = -a[..., 1]
    out['actions'] = a
    out['scan'] = batch['scan'][..., ::-1].copy()
    out['odom'] = batch['odom'] * np.array([1, -1], np.float32)
    out['wheels'] = batch['wheels'][..., ::-1].copy()
    out['current'] = batch['current'][..., ::-1].copy()
    out['pose'] = batch['pose'] * np.array([1, -1, -1], np.float32)
    return out


def raw_batch(data: SessionData, starts: np.ndarray) -> dict:
    """Arrays for windows starting at starts, before standardizing: inputs (B, h+1, d), actions
    (B, h, 2), each target (B, h+1, size), and pose changes (B, h+1, 3)."""
    idx = starts[:, None] + np.arange(data.h + 1)[None, :]
    batch = {
        'inputs': data.inputs[idx],
        'actions': data.actions[idx[:, :-1]],
        'pose': pose_changes(data.pose, idx),
    }
    for name in TARGETS:
        batch[name] = data.targets[name][idx].astype(np.float32)
    return batch


class Normalizer:
    """Standardizes continuous targets with the training windows' mean and spread (their mirror images
    included, so left and right get the same treatment), so no head dominates."""

    def __init__(self, sessions):
        raws = [raw_batch(s, s.train) for s in sessions if len(s.train)]
        raws += [mirror(r) for r in raws]
        self.mean, self.std = {}, {}
        for name, (size, binary) in TARGETS.items():
            if binary:
                continue
            values = np.concatenate([r[name].reshape(-1, size) for r in raws])
            self.mean[name] = values.mean(axis=0).astype(np.float32)
            self.std[name] = np.maximum(values.std(axis=0), 1e-3).astype(np.float32)

    def __call__(self, batch: dict) -> dict:
        out = dict(batch)
        for name in self.mean:
            out[name] = ((batch[name] - self.mean[name]) / self.std[name]).astype(np.float32)
        return out


def window_batch(data: SessionData, starts: np.ndarray, norm: Normalizer, mirrored: bool = False) -> dict:
    """Standardized arrays for windows starting at starts, optionally mirrored left to right."""
    raw = raw_batch(data, starts)
    return norm(mirror(raw) if mirrored else raw)
