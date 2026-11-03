import numpy as np

from pet_learn.replay_check import HORIZONS, compose, replay_starts, true_pose


def test_composing_steps_follows_the_robot_round_a_corner():
    # 1 m ahead, turn left 90 deg, then 1 m ahead (now along +y), all in the frame where each step began.
    deltas = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], [1.0, 0.0, 0.0]]])
    out = compose(deltas)
    assert np.allclose(out[0, -1], [1.0, 1.0, np.pi / 2])


def test_true_pose_is_in_the_frame_of_the_start():
    pose = np.array([[2.0, 3.0, np.pi / 2], [2.0, 4.0, np.pi / 2], [1.0, 4.0, np.pi]])
    out = true_pose(pose, np.array([[0, 1, 2]]))
    assert np.allclose(out[0], [[0, 0, 0], [1, 0, 0], [1, 1, np.pi / 2]])


def test_replay_starts_stop_at_unusable_driving_a_map_reset_and_the_scored_end():
    from test_episodes import drive
    n = 400
    steps = drive([(0.01 * k, 0.0, 0.0) for k in range(n)])
    steps['docked'][150] = True  # unusable at step 150
    steps['slam_reset'] = np.zeros(n, bool)
    steps['slam_reset'][250] = True  # a new SLAM map from step 250
    steps['wheels_enabled'] = np.ones(n, bool)
    steps['wheels_enabled'][380:] = False  # the scored driving ends at 380
    starts, reach = replay_starts(steps, 0, 10)
    by = dict(zip(starts.tolist(), reach.tolist()))
    assert 150 not in by  # an unusable start is skipped
    assert by[100] == 49  # usable up to 149
    assert by[160] == 89  # up to 249, before the map reset
    after = min(k for k in by if k > 250)  # the first start after the reset (the map settles first)
    assert by[after] == 379 - after  # up to the last scored step
    assert by[0] == 149
    assert max(by.values()) <= max(HORIZONS)
