import numpy as np
import pytest
import torch

from pet_learn.dataset import (FLAG_COLUMNS, INPUT_COLUMNS, SCAN_CAP, SENSORS, Normalizer, SessionData, hidden_sensors,
                               hide_in_batch, inputs, mirror_inputs, present, raw_batch, step_targets, window_batch)
from pet_learn.model import WorldModel, current_readings, world_loss


def fake_steps(n=60, ir=True):
    rng = np.random.default_rng(0)
    pose = np.zeros((n, 3))
    pose[:, 0] = np.arange(n) * 0.02  # driving straight ahead
    return {
        't': np.arange(n) / 10.0, 'rate': 10.0,
        'scan': rng.uniform(0.2, 5.0, (n, 60)).astype(np.float32),
        'cmd': np.tile([0.2, 0.0], (n, 1)).astype(np.float32),
        'odom_vel': np.tile([0.2, 0.0], (n, 1)).astype(np.float32),
        'wheel_vels': np.full((n, 2), 5.0, np.float32), 'wheel_current': np.full((n, 2), 180.0, np.float32),
        'mouse_speed': np.full(n, 0.2, np.float32), 'slipping': np.zeros(n, bool), 'stall': np.zeros(n, bool),
        'bump': np.zeros((n, 3), bool), 'docked': np.zeros(n, bool), 'kidnapped': np.zeros(n, bool),
        'ir': (rng.uniform(0, 600, (n, 7)) if ir else np.full((n, 7), np.nan)).astype(np.float32),
        'pose': pose,
    }


def test_ir_is_an_input_and_mirrors_left_to_right():
    steps = fake_steps()
    x = inputs(steps)
    assert x.shape[1] == 87 and np.all(x[:, 81:] == 1)
    ir = x[:, 74:81]
    assert np.allclose(mirror_inputs(x)[:, 74:81], ir[:, ::-1])
    assert np.allclose(mirror_inputs(mirror_inputs(x)), x)


def test_sessions_without_ir_are_masked_out_of_the_loss():
    old = SessionData(fake_steps(ir=False))
    assert np.all(old.inputs[:, 74:82] == 0)  # missing readings and "not recorded"
    norm = Normalizer([old])
    batch = {k: torch.as_tensor(v) for k, v in window_batch(old, old.train[:8], norm).items()}
    _, parts = world_loss(WorldModel(old.inputs.shape[1], z_dim=16, hidden=32), batch)
    assert parts['ir'].item() == 0.0 and all(torch.isfinite(v) for v in parts.values())


def test_a_missing_scan_is_zero_flagged_and_masked_out_of_the_loss():
    steps = fake_steps()
    steps['scan'][30:] = np.nan  # the scans stopped arriving
    data = SessionData(steps)
    x = data.inputs
    assert np.all(x[30:, INPUT_COLUMNS['scan']] == 0) and np.all(x[30:, FLAG_COLUMNS['scan']] == 0)
    assert np.all(x[:30, FLAG_COLUMNS['scan']] == 1) and np.all(x[30:, FLAG_COLUMNS['odom']] == 1)
    assert data.masks['scan_ok'][30:].sum() == 0 and data.masks['scan_ok'][:30].all()
    norm = Normalizer([data])
    windows = np.arange(31, 40)  # entirely without scans
    batch = {k: torch.as_tensor(v) for k, v in window_batch(data, windows, norm).items()}
    _, parts = world_loss(WorldModel(x.shape[1], z_dim=16, hidden=32), batch)
    assert parts['scan'].item() == 0.0 and all(torch.isfinite(v) for v in parts.values())


def test_a_scan_with_nothing_in_range_in_some_sectors_is_still_real():
    steps = fake_steps()
    steps['scan'][:, :10] = 12.0  # the converter puts the lidar's range_max where nothing returned
    data = SessionData(steps)
    assert data.masks['scan_ok'].all() and np.all(data.inputs[:, 0:10] == 1.0)  # far (capped), not missing


def test_each_sensor_sits_in_its_columns_with_its_flag():
    # Catches a change to the order of inputs() that INPUT_COLUMNS / FLAG_COLUMNS (and so hide) miss.
    steps = fake_steps()
    rng = np.random.default_rng(1)
    for key in ('odom_vel', 'wheel_vels', 'wheel_current'):
        steps[key] = rng.uniform(0.1, 1.0, steps[key].shape).astype(np.float32)
    steps['mouse_speed'] = rng.uniform(0.1, 0.3, steps['mouse_speed'].shape).astype(np.float32)
    # Each sensor missing at different steps, so every flag column is different and a swap shows.
    steps['scan'][15:20] = np.nan
    steps['mouse_speed'][20:22] = np.nan
    steps['odom_vel'][25:28] = np.nan
    steps['wheel_vels'][5:10] = np.nan
    steps['wheel_current'][30:33] = np.nan
    steps['ir'][40:45] = np.nan
    x, t, ok = inputs(steps), step_targets(steps), present(steps)
    for name in SENSORS:
        assert np.array_equal(x[:, INPUT_COLUMNS[name]], t[name]), name
        assert np.array_equal(x[:, FLAG_COLUMNS[name]], ok[name][:, 0]), name
    assert len({x[:, c].tobytes() for c in FLAG_COLUMNS.values()}) == len(SENSORS)
    assert ok['wheels'][5:10].sum() == 0 and ok['wheels'][10:].all()
    flags = sorted(FLAG_COLUMNS.values())
    assert flags == list(range(flags[0], x.shape[1]))  # the flags are the last columns, one each


def test_mirroring_swaps_each_sensor_left_to_right():
    steps = fake_steps()
    rng = np.random.default_rng(2)
    steps['odom_vel'][:, 1] = 0.7  # turning left
    steps['wheel_vels'] = rng.uniform(1, 9, steps['wheel_vels'].shape).astype(np.float32)
    steps['wheel_current'] = rng.uniform(50, 400, steps['wheel_current'].shape).astype(np.float32)
    steps['bump'][::3, 0] = True  # bumps on the left only
    steps['cmd'][:, 1] = 0.3  # turning left
    x = inputs(steps)
    m = mirror_inputs(x)
    bumps, previous = slice(67, 70), slice(72, 74)  # inputs(): bump sides, then slip, stall, previous command
    assert np.array_equal(x[:, bumps], steps['bump'].astype(np.float32))
    assert np.array_equal(m[:, bumps], x[:, bumps][:, ::-1])
    assert np.allclose(m[1:, previous], x[1:, previous] * [1, -1]) and np.any(x[1:, 73] != 0)
    for name in ('scan', 'wheels', 'current', 'ir'):
        assert np.allclose(m[:, INPUT_COLUMNS[name]], x[:, INPUT_COLUMNS[name]][:, ::-1]), name
    odom = INPUT_COLUMNS['odom']
    assert np.allclose(m[:, odom], x[:, odom] * [1, -1])
    assert np.array_equal(m[:, sorted(FLAG_COLUMNS.values())], x[:, sorted(FLAG_COLUMNS.values())])


def test_val_and_test_windows_dont_overlap_training_or_each_other():
    data = SessionData(fake_steps(n=400))
    h = data.h
    assert len(data.val) and len(data.test)
    assert data.train.max() + h < data.val.min() and data.val.max() + h < data.test.min()


def test_a_session_ending_without_scans_still_holds_out_driving():
    steps = fake_steps(n=1000)
    steps['scan_age'] = np.zeros(1000, np.float32)
    steps['scan_age'][400:] = np.arange(600) / 10.0 + 0.1  # the lidar's messages stop at step 400
    steps['scan'][400:] = np.nan
    data = SessionData(steps)
    assert len(data.val) and len(data.test)  # split over the scored driving, not all 1000 steps
    assert data.test_split < 400 + 100  # within 10 s of the last scan
    assert data.test.max() + data.h < 400 + 101  # and test ends with the scored driving


def test_hidden_sensors_are_missing_to_the_encoder_but_still_targets():
    data = SessionData(fake_steps())
    raw = raw_batch(data, data.train[:6])
    hidden = {name: np.zeros(6, bool) for name in INPUT_COLUMNS}
    hidden['scan'][:3] = True
    out = hide_in_batch(raw, hidden)
    assert np.all(out['inputs'][:3, :, INPUT_COLUMNS['scan']] == 0) and np.all(out['inputs'][:3, :, FLAG_COLUMNS['scan']] == 0)
    assert np.array_equal(out['inputs'][3:], raw['inputs'][3:])
    assert np.array_equal(out['scan'], raw['scan']) and out['scan_ok'].all()  # still learned from
    assert np.all(out['scan_seen'][:3] == 0) and np.all(out['scan_seen'][3:] == 1)
    # Predictions don't start from the reading the encoder couldn't see.
    t = {k: torch.as_tensor(v) for k, v in out.items()}
    assert torch.all(current_readings(t)['scan'][:3] == 0) and torch.equal(current_readings(t)['scan'][3:], t['scan'][3:, :1])
    assert hidden_sensors(1000, np.random.default_rng(0), 0.1)['scan'].mean() == pytest.approx(0.1, abs=0.03)
