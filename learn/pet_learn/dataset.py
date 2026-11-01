"""Training data for the driving model (see MODEL.md): per-step inputs and actions, prediction targets,
and windows of consecutive steps for predicting up to H steps ahead."""
import numpy as np

from pet_learn.episodes import scored, usable

FORWARD_CAP, TURN_CAP = 0.3, 1.5  # the safety layer's caps (m/s, rad/s): actions are scaled by them
SCAN_CAP = 4.0  # lidar sectors are capped at this (m): beyond it, distance matters little for driving
H = 10  # steps ahead to predict (1 s at 10 Hz)
POSE_SCALE = (0.3, 0.3, 1.0)  # pose change targets (m, m, rad) are divided by these: about 1 s of driving
IR_SCALE = np.log1p(1000.0)  # IR readings go in as log(1 + reading) / this: 0-5 is nothing near, ~500 a hand at 5 cm

# Prediction targets: name -> (size, binary?)
TARGETS = {
    'scan': (60, False),
    'floor': (1, False),
    'odom': (2, False),
    'wheels': (2, False),
    'current': (2, False),
    'bump': (1, True),
    'slip': (1, True),
    'ir': (7, False),
}
# Readings that can be missing: their messages stopped arriving (the lidar's often do first, over
# Wi-Fi), or weren't recorded (IR, before it was relayed). Each has a batch key for a (n, 1) mask that is
# 1 where the reading is real; missing readings are 0, as targets and as inputs, and the encoder sees
# the masks too, so it can learn to drive on what's left.
SENSORS = ('scan', 'floor', 'odom', 'wheels', 'current', 'ir')
MASKS = {name: f'{name}_ok' for name in SENSORS}
# Where each sensor's readings and its "real" flag sit in the encoder's inputs (inputs()).
INPUT_COLUMNS = {'scan': slice(0, 60), 'floor': slice(60, 61), 'odom': slice(61, 63), 'wheels': slice(63, 65),
                 'current': slice(65, 67), 'ir': slice(74, 81)}
FLAG_COLUMNS = {'ir': 81, 'scan': 82, 'floor': 83, 'odom': 84, 'wheels': 85, 'current': 86}


def present(steps: dict) -> dict:
    """Where each sensor's readings are real: name -> (n, 1) float32, 1 if real. A missing scan is all nan
    (a real one has no nan: sectors with nothing in range hold the lidar's range_max)."""
    ok = {
        'scan': ~np.isnan(steps['scan']).all(axis=1),
        'floor': np.isfinite(steps['mouse_speed']),
        'odom': np.isfinite(steps['odom_vel']).all(axis=1),
        'wheels': np.isfinite(steps['wheel_vels']).all(axis=1),
        'current': np.isfinite(steps['wheel_current']).all(axis=1),
        'ir': np.isfinite(ir_readings(steps)).all(axis=1),
    }
    return {name: v[:, None].astype(np.float32) for name, v in ok.items()}


def step_targets(steps: dict) -> dict:
    """Each step's readings, as targets: name -> (n, size) float32, in the units the inputs use; 0 where
    missing (present())."""
    scan_ok = present(steps)['scan']
    return {
        'scan': scan_ok * np.nan_to_num(np.clip(steps['scan'], 0, SCAN_CAP), nan=SCAN_CAP) / SCAN_CAP,
        'floor': np.nan_to_num(np.clip(steps['mouse_speed'], 0, 1.0))[:, None] / FORWARD_CAP,
        'odom': np.nan_to_num(steps['odom_vel'] / np.array([FORWARD_CAP, TURN_CAP], np.float32)),
        'wheels': np.nan_to_num(steps['wheel_vels'] / 10.0),  # rad/s; 0.3 m/s is about 8 rad/s
        'current': np.nan_to_num(steps['wheel_current'] / 1000.0),  # A
        'bump': steps['bump'].any(axis=1)[:, None].astype(np.float32),
        'slip': steps['slipping'][:, None].astype(np.float32),
        'ir': np.nan_to_num(np.log1p(np.clip(ir_readings(steps), 0, 4000)) / IR_SCALE, nan=0.0).astype(np.float32),
    }


def ir_readings(steps: dict) -> np.ndarray:
    """(n, 7) IR readings, left to right; nan where not recorded."""
    if 'ir' in steps:
        return steps['ir']
    return np.full((len(steps['t']), 7), np.nan, np.float32)


def actions(steps: dict) -> np.ndarray:
    return (steps['cmd'] / np.array([FORWARD_CAP, TURN_CAP], np.float32)).astype(np.float32)


def inputs(steps: dict) -> np.ndarray:
    """What the encoder sees at each step: every target reading, plus wheel stall, the previous drive
    command, and whether each sensor's readings are real (FLAG_COLUMNS)."""
    t = step_targets(steps)
    ok = present(steps)
    previous = np.vstack([np.zeros((1, 2), np.float32), actions(steps)[:-1]])
    stall = steps['stall'][:, None].astype(np.float32)
    bump_sides = steps['bump'].astype(np.float32)  # left, centre, right: more than the target's any-bump
    parts = [t['scan'], t['floor'], t['odom'], t['wheels'], t['current'], bump_sides, t['slip'], stall, previous, t['ir'],
             ok['ir'], ok['scan'], ok['floor'], ok['odom'], ok['wheels'], ok['current']]
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
    """One converted session's model data, with windows split by time: training (the first 80%), val
    (the next 10%: for picking the best model and when to stop) and test (the last 10%: for reporting
    how good it is, on driving the choice never saw). The percentages are of the session's scored
    driving (episodes.scored), not of all its steps: a session can end in a long stretch without
    scans (or on the dock), and cutting by step count would put its whole held-out part there. The
    split steps are kept (split, test_split) so the driver holds out the same driving."""

    def __init__(self, steps: dict, holdout: float = 0.2, h: int = H):
        self.steps = steps
        self.h = h
        self.inputs = inputs(steps)
        self.actions = actions(steps)
        self.targets = step_targets(steps)
        self.masks = {MASKS[name]: ok for name, ok in present(steps).items()}
        self.pose = steps['pose'].astype(np.float64)
        ok = usable(steps)
        n = len(ok)
        # A window starting at t needs steps t..t+h all usable.
        bad = np.concatenate([[0], np.cumsum(~ok)])
        starts = np.flatnonzero(bad[np.arange(n - h) + h + 1] - bad[np.arange(n - h)] == 0)
        u = np.flatnonzero(scored(steps))
        if len(u) == 0:  # nothing to hold out
            split = test_split = end = n
        else:
            split, test_split = (int(u[min(int(len(u) * f), len(u) - 1)]) for f in (1 - holdout, 1 - holdout / 2))
            end = int(u[-1]) + 1  # test ends with the scored driving, like the driver's
        self.split, self.test_split = split, test_split
        self.train = starts[starts + h < split]
        self.val = starts[(starts >= split) & (starts + h < test_split)]
        self.test = starts[(starts >= test_split) & (starts + h < end)]


def mirror_inputs(x: np.ndarray) -> np.ndarray:
    """Encoder inputs (..., d) flipped left to right (see mirror)."""
    x = x.copy()
    # Input layout (inputs()): scan 60, floor 1, odom 2, wheels 2, current 2, bump sides 3, slip, stall, previous 2,
    # IR 7, then the sensors' "real" flags (FLAG_COLUMNS), which don't change in a mirror
    x[..., 0:60] = x[..., 59::-1]
    x[..., 62] = -x[..., 62]  # turn rate
    x[..., 63:65] = x[..., 64:62:-1]  # wheels left <-> right
    x[..., 65:67] = x[..., 66:64:-1]  # currents
    x[..., 67:70] = x[..., 69:66:-1]  # bump left <-> right
    x[..., 73] = -x[..., 73]  # previous turn command
    x[..., 74:81] = x[..., 80:73:-1]  # IR: the sensors' angles are symmetric about straight ahead
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
    out['ir'] = batch['ir'][..., ::-1].copy()
    return out


def raw_batch(data: SessionData, starts: np.ndarray) -> dict:
    """Arrays for windows starting at starts, before standardizing: inputs (B, h+1, d), actions
    (B, h, 2), each target (B, h+1, size), pose changes (B, h+1, 3), and for each sensor in SENSORS
    whether the reading is real (<name>_ok) and whether the encoder sees it (<name>_seen: the same,
    until hide() hides it), each (B, h+1, 1)."""
    idx = starts[:, None] + np.arange(data.h + 1)[None, :]
    batch = {
        'inputs': data.inputs[idx],
        'actions': data.actions[idx[:, :-1]],
        'pose': pose_changes(data.pose, idx),
    }
    for name in TARGETS:
        batch[name] = data.targets[name][idx].astype(np.float32)
    for key, mask in data.masks.items():
        batch[key] = mask[idx]
    for name in SENSORS:
        batch[f'{name}_seen'] = batch[MASKS[name]].copy()
    return batch


def hidden_sensors(n: int, rng, p: float) -> dict:
    """For n examples, which sensors to hide in training: each sensor in each example with probability
    p. Real readings go missing (the lidar's over Wi-Fi, mostly), and hiding them teaches the model to
    get by on the rest."""
    return {name: rng.random(n) < p for name in SENSORS}


def hide(x: np.ndarray, hidden: dict) -> np.ndarray:
    """Encoder inputs x (n, ..., d) with the hidden sensors' readings and flags zeroed, as if missing."""
    x = x.copy()
    for name, rows in hidden.items():
        x[rows, ..., INPUT_COLUMNS[name]] = 0.0
        x[rows, ..., FLAG_COLUMNS[name]] = 0.0
    return x


def hide_in_batch(batch: dict, hidden: dict) -> dict:
    """A raw window batch with sensors hidden from the encoder for whole windows. The targets stay:
    the model learns to predict what it can't see."""
    out = dict(batch)
    out['inputs'] = hide(batch['inputs'], hidden)
    for name, rows in hidden.items():
        seen = batch[f'{name}_seen'].copy()
        seen[rows] = 0.0
        out[f'{name}_seen'] = seen
    return out


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
            if name in MASKS:
                values = values[np.concatenate([r[MASKS[name]].reshape(-1) for r in raws]) > 0.5]
            if len(values) == 0:  # no session has this reading yet
                self.mean[name], self.std[name] = np.zeros(size, np.float32), np.ones(size, np.float32)
                continue
            self.mean[name] = values.mean(axis=0).astype(np.float32)
            self.std[name] = np.maximum(values.std(axis=0), 1e-3).astype(np.float32)

    def __call__(self, batch: dict) -> dict:
        out = dict(batch)
        for name in self.mean:
            out[name] = ((batch[name] - self.mean[name]) / self.std[name]).astype(np.float32)
        return out


def window_batch(data: SessionData, starts: np.ndarray, norm: Normalizer, mirrored: bool = False,
                 rng=None, hide_p: float = 0.0) -> dict:
    """Standardized arrays for windows starting at starts, optionally mirrored left to right, and with
    sensors hidden at random (hide_p, hidden_sensors)."""
    raw = raw_batch(data, starts)
    if hide_p > 0:
        raw = hide_in_batch(raw, hidden_sensors(len(starts), rng, hide_p))
    return norm(mirror(raw) if mirrored else raw)
