import numpy as np
import torch

from pet_learn.dataset import Normalizer, SessionData, inputs, mirror_inputs, window_batch
from pet_learn.model import WorldModel, world_loss


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
    assert x.shape[1] == 82 and np.all(x[:, 81] == 1)
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
