import numpy as np

from pet_learn.episodes import evaluate, hindsight_episodes, usable
from pet_learn.geodesic import GridMap
from pet_learn.reward import RewardParams


class OneMap:
    def __init__(self, gmap):
        self.gmap = gmap

    def map_at(self, t):
        return self.gmap


def drive(poses, rate=10.0):
    """Steps for a robot that drove through poses (x, y, yaw), one per step, at 0.2 A."""
    n = len(poses)
    return {
        'rate': np.float64(rate), 't': np.arange(n) / rate, 'pose': np.array(poses, dtype=np.float32),
        'docked': np.zeros(n, bool), 'kidnapped': np.zeros(n, bool),
        'wheel_current': np.full((n, 2), 100.0, dtype=np.float32), 'bump': np.zeros((n, 3), bool),
        'cmd': np.tile([0.2, 0.0], (n, 1)).astype(np.float32), 'mouse_speed': np.full(n, 0.2, np.float32),
    }


OPEN = OneMap(GridMap(np.zeros((60, 60), np.int8), (0.0, 0.0), 0.05))  # 3 m x 3 m, all free


def test_a_straight_drive_scores_close_to_perfect():
    steps = drive([(0.5 + 0.02 * k, 1.5, 0.0) for k in range(101)])  # 2 m east at 0.2 m/s
    e = evaluate(steps, OPEN, 0, 100, (2.5, 1.5, 0.0))
    assert e.success and e.end < 100  # arrives within goal_radius before the last step
    assert 0.9 < e.spl <= 1.0
    assert e.reward > 0


def test_a_detour_scores_lower():
    out = [(0.5 + 0.02 * k, 1.5 + 0.02 * k, 0.0) for k in range(50)]
    back = [(1.5 + 0.02 * k, 2.5 - 0.02 * k, 0.0) for k in range(51)]
    detour = evaluate(drive(out + back), OPEN, 0, 100, (2.5, 1.5, 0.0))
    straight = evaluate(drive([(0.5 + 0.02 * k, 1.5, 0.0) for k in range(101)]), OPEN, 0, 100, (2.5, 1.5, 0.0))
    assert detour.success and detour.spl < straight.spl and detour.reward < straight.reward


def test_wrong_heading_is_not_arrival():
    steps = drive([(0.5 + 0.02 * k, 1.5, 0.0) for k in range(101)])
    e = evaluate(steps, OPEN, 0, 100, (2.5, 1.5, np.pi / 2))
    assert not e.success and e.spl == 0.0


def test_pose_jumps_and_pickups_are_not_usable():
    steps = drive([(0.5, 1.5, 0.0), (0.52, 1.5, 0.0), (1.5, 1.5, 0.0), (1.52, 1.5, 0.0)])
    steps['kidnapped'][3] = True
    assert usable(steps).tolist() == [True, True, False, False]


def test_hindsight_episodes_stay_within_usable_steps():
    steps = drive([(0.5 + 0.01 * k, 1.5, 0.0) for k in range(200)])
    steps['kidnapped'][100:110] = True
    episodes = hindsight_episodes(steps, OPEN, 20, np.random.default_rng(0), min_s=3, max_s=8)
    assert episodes
    for e in episodes:
        assert not (e.start < 110 and e.end >= 100)


def test_a_rescue_is_not_usable_until_the_robot_is_set_down():
    steps = drive([(0.5, 1.5, 0.0)] * 300)
    steps['rescue'] = np.zeros(300, bool)
    steps['rescue'][100] = True
    steps['kidnapped'][120:150] = True  # lifted out 2 s after the press, set down 3 s later
    steps['rescue'][250] = True  # freed without lifting: 10 s
    ok = usable(steps)
    assert ok[99] and not ok[100:150].any() and ok[150]
    assert not ok[250:350].any() and ok[249]


def test_episodes_that_run_into_a_rescue_end_as_failures():
    from pet_learn.goals import relabel
    steps = drive([(0.5 + 0.005 * k, 1.5, 0.0) for k in range(300)])
    steps['rescue'] = np.zeros(300, bool)
    steps['rescue'][150] = True
    steps['kidnapped'][160:170] = True
    ok = usable(steps)
    t = relabel(steps, OPEN, ok, 0, 300, 0, np.random.default_rng(0), rescue_goals=20)
    assert len(t.step) and not t.hindsight.any()
    last = t.step == 148  # the last move, into the step before the press
    assert last.any() and t.done[last].all()
    assert (t.reward[last] < -RewardParams().rescue_penalty + 0.5).all()
    assert (t.step < 150).all()
