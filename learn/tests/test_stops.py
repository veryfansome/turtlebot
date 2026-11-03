import math

import numpy as np

from pet_learn.goal_check import goal_parts, interval, pair_states
from pet_learn.stops import find_rests, stop_pairs

RATE = 10.0


def drive(plan):
    """Steps from a plan of (seconds, forward m/s, turn rad/s, command forward, command turn): the wheels
    do the first two, the stick says the last two (a stall: told to drive, not moving). The pose follows
    the wheels."""
    rows = np.concatenate([np.tile(row, (int(round(row[0] * RATE)), 1)) for row in plan])
    odom, cmd = rows[:, 1:3], rows[:, 3:5]
    pose = np.zeros((len(rows), 3))
    for k in range(1, len(rows)):
        x, y, yaw = pose[k - 1]
        f, w = odom[k - 1]
        pose[k] = x + f * math.cos(yaw) / RATE, y + f * math.sin(yaw) / RATE, yaw + w / RATE
    n = len(rows)
    return {'rate': np.array(RATE), 't': np.arange(n) / RATE, 'odom_vel': odom, 'cmd': cmd, 'pose': pose,
            'bump': np.zeros((n, 3), bool)}


GO = (2, 0.2, 0.0, 0.2, 0.0)  # driving
STOP = (1, 0.0, 0.0, 0.0, 0.0)  # let go, standing still
TURN = (1, 0.0, 1.0, 0.0, 1.0)  # turning in place
PUSH = (1, 0.0, 0.0, 0.2, 0.0)  # told to drive, not moving


def kinds(steps, ok=None):
    ok = np.ones(len(steps['t']), bool) if ok is None else ok
    return [r.kind for r in find_rests(steps, ok)]


def test_stops_are_classified_by_how_the_robot_came_to_rest():
    steps = drive([STOP, GO, STOP, TURN, STOP, GO, PUSH, GO, STOP])
    assert kinds(steps) == ['not from motion', 'intended', 'intended', 'pushing', 'intended']
    bumped = dict(steps, bump=steps['bump'].copy())
    bumped['bump'][28, 1] = True  # 0.2 s before the first stop after driving
    assert kinds(bumped)[1] == 'bump'
    late = dict(steps, bump=steps['bump'].copy())
    late['bump'][34, 0] = True  # 0.4 s after it: the stop came first
    assert kinds(late)[1] == 'intended'
    ok = np.ones(len(steps['t']), bool)
    ok[29] = False  # a gap in scored driving just before the stop: how it got there isn't known
    assert kinds(steps, ok)[1] == 'not from motion'


def test_a_single_step_stop_counts_and_missing_odometry_is_not_rest():
    steps = drive([GO, (0.1, 0.0, 0.0, 0.0, 0.0), GO])
    rests = find_rests(steps, np.ones(len(steps['t']), bool))
    assert [(r.kind, r.start, r.end) for r in rests] == [('intended', 20, 20)]
    steps['odom_vel'][20] = np.nan
    assert find_rests(steps, np.ones(len(steps['t']), bool)) == []


def test_pairs_run_from_any_rest_to_later_intended_stops_within_the_window():
    steps = drive([STOP, GO, STOP, GO, STOP])
    ok = np.ones(len(steps['t']), bool)
    starts = [r.end for r in find_rests(steps, ok)]  # 9, 39, 69
    goals = [r.start for r in find_rests(steps, ok) if r.kind == 'intended']  # 30, 60
    assert stop_pairs(steps, ok, 0, len(ok)) == [(starts[0], goals[0]), (starts[0], goals[1]), (starts[1], goals[1])]
    assert stop_pairs(steps, ok, 0, len(ok), window_s=3.0) == [(starts[0], goals[0]), (starts[1], goals[1])]
    assert stop_pairs(steps, ok, 0, 50) == [(starts[0], goals[0])]  # the region ends before the second goal
    ok[45] = False  # unscored driving between the second stop and the third
    assert stop_pairs(steps, ok, 0, len(ok)) == [(starts[0], goals[0])]
    reset = dict(steps, slam_reset=np.zeros(len(ok), bool))
    reset['slam_reset'][15] = True  # a fresh SLAM map during the first drive
    assert stop_pairs(reset, np.ones(len(ok), bool), 0, len(ok)) == [(starts[1], goals[1])]


def test_the_goal_part_of_a_turn_is_what_mirroring_the_goal_changes():
    side = np.array([1.0, 1.0, -1.0, 1.0])
    turn = np.array([0.5, 0.3, -0.4, 0.2])
    mirrored = np.array([-0.5, 0.3, 0.0, -0.1])  # follows the goal; ignores it; half; reverses
    goal, state, flips = goal_parts(turn, mirrored, side)
    assert np.allclose(goal, [0.5, 0.0, 0.2, 0.15]) and np.allclose(state, [0.0, 0.3, -0.2, 0.05])
    assert flips.tolist() == [True, False, True, True]


def test_states_near_the_goal_take_their_side_from_the_heading():
    steps = drive([STOP, GO, STOP, TURN, STOP])
    steps['pose'][:, :2] += 1.0
    st = pair_states(steps, [(0, 30), (39, 50)])
    near = st['pair'] == 1  # turning in place to face a heading to the left: within the arrival radius
    assert (st['distance'][near] < 0.2).all() and (st['side'][near] == 1).all()
    far = st['pair'] == 0  # the first drive's goal is dead ahead
    assert np.allclose(st['bearing'][far][:5], 0, atol=1e-6) and (st['doing'][far][0] == 0)


def test_intervals_resample_clusters_not_states():
    rng = np.random.default_rng(0)
    values = np.r_[np.zeros(1000), np.ones(10)]
    mean, lo, hi = interval(values, np.r_[np.zeros(1000), np.ones(10)].astype(int), rng)
    assert math.isclose(mean, 10 / 1010) and lo == 0.0 and hi == 1.0  # two clusters: either could be all there is


def test_a_wobble_with_the_stick_released_doesnt_split_a_rest():
    wobble = (0.1, 0.0, 0.03, 0.0, 0.0)  # one step of odometry settling, nothing commanded
    steps = drive([GO, STOP, wobble, STOP, GO])
    rests = find_rests(steps, np.ones(len(steps['t']), bool))
    assert [(r.kind, r.start, r.end) for r in rests] == [('intended', 20, 40)]
    pushed = drive([GO, PUSH, wobble, STOP, GO])  # a stall, then the odometry settles: still not a goal
    assert kinds(pushed) == ['pushing']
    tapped = drive([GO, STOP, (0.3, 0.1, 0.0, 0.1, 0.0), STOP, GO])  # a real nudge between two stops
    assert kinds(tapped) == ['intended', 'intended']


def test_turning_states_and_goals_on_the_right():
    steps = drive([STOP, (2, 0.2, -0.5, 0.2, -0.5), STOP])  # an arc to the right
    st = pair_states(steps, [(9, 30)])
    assert (st['side'][st['distance'] > 0.2] == -1).all()
    assert st['doing'][0] == 0 and (st['doing'][2:15] == 2).all()  # at rest, then the previous command turned


def test_pairs_end_at_their_first_arrival_as_training_episodes_do():
    steps = drive([STOP, GO, (2, 0.0, 0.0, 0.0, 0.0)])  # drive 0.4 m, then sit
    st = pair_states(steps, [(9, 45)])  # a goal taken while sitting
    arrived_at = int(np.flatnonzero(st['after'])[0])
    within = np.flatnonzero(st['distance'] <= 0.2)[0]  # the reward's arrival: within 0.2 m (heading matches)
    assert arrived_at == within and 10 < st['step'][arrived_at] < 30  # while still driving in
    assert st['after'][arrived_at:].all() and not st['after'][:arrived_at].any()


def test_a_nudge_in_a_rests_last_moment_keeps_two_stops_apart():
    nudge = (0.1, 0.0, 0.0, 0.1, 0.0)  # the stick pushed during the rest's last step ...
    lagged = (0.2, 0.08, 0.0, 0.0, 0.0)  # ... reaching the wheels after it, as the base lags
    steps = drive([GO, STOP, nudge, lagged, STOP, GO])
    assert kinds(steps) == ['intended', 'intended']


def test_the_release_may_come_up_to_0_2_s_before_the_wheels_stop():
    coast = lambda s: (s, 0.1, 0.0, 0.0, 0.0)  # let go, still rolling
    assert kinds(drive([GO, coast(0.2), STOP, GO])) == ['intended']
    assert kinds(drive([GO, coast(0.5), STOP, GO])) == ['intended']  # let go earlier and rolled to a stop: still a let-go
    late = (0.2, 0.1, 0.0, 0.1, 0.0)  # still told to drive 0.1 s before the stop ...
    assert kinds(drive([GO, late, (0.1, 0.0, 0.0, 0.1, 0.0), STOP, GO])) == ['pushing']  # ... and at it: not let go


def test_rest_merging_needs_scored_steps_and_bumps_just_after_count():
    wobble = (0.1, 0.0, 0.03, 0.0, 0.0)
    steps = drive([GO, STOP, wobble, STOP, GO])
    ok = np.ones(len(steps['t']), bool)
    ok[30] = False  # the wobble step isn't scored
    assert kinds(steps, ok) == ['intended', 'not from motion']
    late = drive([GO, STOP, GO])
    late['bump'][21, 2] = True  # 0.1 s after the robot came to rest
    assert kinds(late) == ['bump']


def test_short_stops_can_start_episodes_but_not_end_them_when_a_minimum_is_set():
    from pet_learn.stops import StopParams
    brief = (0.1, 0.0, 0.0, 0.0, 0.0)
    steps = drive([STOP, GO, brief, GO, STOP])
    ok = np.ones(len(steps['t']), bool)
    assert len(stop_pairs(steps, ok, 0, len(ok))) == 3
    longer = stop_pairs(steps, ok, 0, len(ok), p=StopParams(min_rest_s=0.5))
    assert longer == [(9, 51), (30, 51)]  # the 0.1 s stop starts an episode, but isn't a goal


def test_pairs_already_at_their_goal_are_left_out():
    hop = (0.5, 0.2, 0.0, 0.2, 0.0)  # 0.1 m: still within the arrival radius
    steps = drive([STOP, hop, STOP])
    assert stop_pairs(steps, np.ones(len(steps['t']), bool), 0, len(steps['t'])) == []


def test_reporting_helpers(monkeypatch):
    from pet_learn import stops
    from pet_learn.stops import after_stop, bump_timing, coverage
    monkeypatch.setattr(stops, 'scored', lambda steps: np.ones(len(steps['t']), bool))
    import os, tempfile
    steps = drive([STOP, GO, STOP, (2, -0.2, 0.0, -0.2, 0.0)])
    ok = np.ones(len(steps['t']), bool)
    rests = find_rests(steps, ok)
    assert after_stop(steps, rests[1]) == 'reverses' and after_stop(steps, rests[0]) == 'sets off'
    turned = drive([GO, STOP, TURN])
    assert after_stop(turned, find_rests(turned, np.ones(len(turned['t']), bool))[0]) == 'turns'
    assert after_stop(drive([GO, STOP, GO]), find_rests(drive([GO, STOP, GO]), np.ones(50, bool))[0]) == 'drives on'
    from pet_learn.stops import arrival
    s, g = rests[0].end, rests[1].start
    assert math.isclose(coverage(steps, ok, [(s, g)]), (arrival(steps['pose'], s, g) - s + 1) / len(ok))
    bumped = drive([STOP, (4, 0.2, 0.0, 0.2, 0.0), STOP])
    bumped['bump'][47, 0] = True  # 0.3 s before the robot comes to rest at step 50
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 's.npz')
        np.savez(path, **bumped)
        per_step, _ = bump_timing([path], span_s=1.0)
    assert per_step[2] == 100 and per_step.sum() == 100  # one onset, three steps before the one rest


def test_check_keeps_states_after_arrival_out_of_every_other_row():
    from pet_learn.goal_check import check
    steps = drive([STOP, GO, (2, 0.0, 0.0, 0.0, 0.0)])
    st = pair_states(steps, [(9, 45)])
    n = len(st['step'])
    policy = np.where(st['after'][:, None], 9.0, 0.1) * np.ones((n, 2))
    r = check(st, policy, -policy, np.zeros((n, 2)), np.random.default_rng(0))
    assert r['after']['states'] == st['after'].sum() and r['approach']['after arriving']['policy_forward'] == 9.0
    rows = [r['far'], r['at_goal'], r['stopped'], r['straight'], r['turning']] + list(r['bearing'].values())
    assert sum(x['states'] for x in rows[:2]) == n - st['after'].sum()
    assert all(not (x['goal_turn'][0] > 1) for x in rows if x['states'])
    assert all(v['policy_forward'] != 9.0 for k, v in r['approach'].items() if k != 'after arriving' and v['states'])


def test_arrival_needs_the_heading_too_and_lasts_once_reached():
    through = drive([STOP, (3, 0.2, 0.0, 0.2, 0.0)])  # drive through the goal and out the far side
    st = pair_states(through, [(9, 20)])
    assert st['after'][-1] and st['distance'][-1] <= 0.2
    st = pair_states(through, [(9, 39)])
    first = int(np.flatnonzero(st['after'])[0])
    assert st['after'][first:].all()  # stays arrived even if it drives past
    turned = drive([STOP, (1, 0.0, 0.0, 0.0, 0.0), (1.5, 0.0, 1.0, 0.0, 1.0), STOP])  # a turn in place to face left
    st = pair_states(turned, [(9, 34)])
    heading = np.degrees(np.abs(np.arctan2(st['goal'][:, 3], st['goal'][:, 4])))
    assert not st['after'][heading > 20].any() and st['after'][heading < 19].all()


def test_the_doing_code_uses_the_previous_command():
    steps = drive([STOP, (2, 0.2, -0.5, 0.2, -0.5), STOP])
    st = pair_states(steps, [(9, 30)])
    assert st['doing'][1] == 1  # step 10 is told to turn, but the policy sees step 9's command: none
