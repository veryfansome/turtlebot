"""Where the person driving brought the robot to rest: natural starts and goals for driving episodes.

Today's hindsight goals (pet_learn.goals) are poses at random steps, mostly taken while the robot was
moving (about 70% faster than 0.05 m/s, forward or back, in a sample of 3,000, 2026-11-02), and
the episodes leading to them mostly start moving too. A car's natural episode runs from parked to parked. Here:

- the robot is at rest when the base's odometry (/odom, which follows the body: it reads about zero
  while the wheels slip) says it isn't moving or turning: under rest_lin and rest_ang. Parked, it
  reads at most about 0.0006 m/s and 0.005 rad/s, so these sit well above its noise; at 10 Hz they
  are about 1 mm and 0.1 degree per step, far inside the reward's 0.2 m and 20 degree arrival test.
  Choosing them is choosing how slow a robot that's still braking must be to count as stopped, and the
  counts move with them (`--sensitivity` prints how). The commands can't say it: they read zero while
  the robot still coasts (up to 0.2 m/s in 90% of such steps, 0.31 in 99%);
- a rest is reached from motion if the step before it wasn't at rest: a rest after a gap in the data,
  or at the start, isn't. Two rests are one, of the first one's kind, when nothing told the robot to
  move between them: the command stayed zero in between, and no command started in the first rest's
  last release_s (the base's lag would show it only afterwards). What moved in between was mostly the
  odometry settling (on the 6 sessions, 2026-11-02, a median peak of 0.005 m/s), sometimes coasting
  or a push; a command that was already on as the robot came to rest (pushing against something)
  doesn't keep the rests apart;
- an intended stop is a rest reached from motion where the person let go: the command (/cmd_vel,
  which also reads zero when none arrived for 0.5 s, convert.COMMAND_TIMEOUT) was exactly zero on both
  axes at the stop or within release_s before it, with no bump from bump_before_s before it to
  bump_after_s after. release_s covers the base's lag: the wheels follow the command about 0.2 s
  later. Bumps start about 5-6 times as often as background in each 0.1 s up to 0.6 s before a rest,
  about twice background from 0.7 to 1.1 s, and at background from about 1.2 s on (2026-11-02;
  printed, with the counts for wider windows): bump_before_s is the sharp edge. A stop while still told to drive or turn is pushing or stalled against something, and a stop
  at a bump is a collision: neither is a goal. A turn in place that stops is an intended stop too: its
  goal is the heading it faced;
- every rest, intended or not, can start an episode: the robot sets off from standing still.

An intended stop is a goal if it lasts at least min_rest_s (every rest can still start an episode); the
default, a single 0.1 s step, is the owner's choice (2026-11-02): the slowing into it is the person's. Most stops are that brief, and the robot often
drives off again at once, sometimes the other way (a reversal): `uv run python -m pet_learn.stops`
prints the counts, the lengths, what followed, and how much of the scored driving the pairs cover.
"""
from dataclasses import dataclass, replace
import glob
import os

import numpy as np

from pet_learn.episodes import map_segment, scored
from pet_learn.reward import RewardParams, arrived, heading_error

KINDS = ('intended', 'bump', 'pushing', 'not from motion')


@dataclass(frozen=True)
class StopParams:
    rest_lin: float = 0.01  # m/s: slower than this is standing still ...
    rest_ang: float = 0.02  # rad/s: ... and turning slower than this
    min_rest_s: float = 0.1  # an intended stop is a goal if it lasts at least this long (one step: any)
    release_s: float = 0.2  # the person let go: the command was exactly zero at the stop or within this before
    bump_before_s: float = 0.6  # a bump this long before the stop ...
    bump_after_s: float = 0.2  # ... or this long after makes it a collision


@dataclass(frozen=True)
class Rest:
    start: int  # first step at rest: the pose an episode that stops here reaches
    end: int  # last step at rest: where an episode setting off from here starts
    kind: str  # one of KINDS


def runs_of(mask: np.ndarray) -> list:
    """(start, stop) of each run of True, stop exclusive."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def at_rest(steps: dict, p: StopParams = StopParams()) -> np.ndarray:
    """(n,) the odometry says the robot is standing still (missing odometry: not at rest)."""
    v = steps['odom_vel']
    with np.errstate(invalid='ignore'):
        return (np.abs(v[:, 0]) < p.rest_lin) & (np.abs(v[:, 1]) < p.rest_ang)


def find_rests(steps: dict, ok: np.ndarray = None, p: StopParams = StopParams()) -> list:
    """Every rest within scored driving (ok, by default episodes.scored), classified (KINDS)."""
    rate = float(steps['rate'])
    ok = scored(steps) if ok is None else ok
    release = max(1, int(round(p.release_s * rate)))
    before, after = (int(round(s * rate)) for s in (p.bump_before_s, p.bump_after_s))
    odom_ok = np.isfinite(steps['odom_vel']).all(axis=1)
    released = (steps['cmd'] == 0).all(axis=1)
    starts = ~released & np.r_[True, released[:-1]]  # a command begins
    bump = steps['bump'].any(axis=1) if 'bump' in steps else np.zeros(len(ok), bool)
    runs = []
    for a, b in runs_of(at_rest(steps, p) & ok):
        # Joined to the rest before when nothing told the robot to move in between: the command stayed zero,
        # no command began in that rest's last release steps (the base's lag), and every step was scored.
        if (runs and released[runs[-1][1]:a].all() and not starts[max(0, runs[-1][1] - release):runs[-1][1]].any()
                and ok[runs[-1][1]:a].all()):
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    rests = []
    for a, b in runs:
        if a == 0 or not ok[a - 1] or not odom_ok[a - 1]:  # nothing known about how it got here
            kind = 'not from motion'
        elif bump[max(0, a - before):a + after + 1].any():
            kind = 'bump'
        elif released[max(0, a - release):a + 1].any():
            kind = 'intended'
        else:
            kind = 'pushing'
        rests.append(Rest(a, b - 1, kind))
    return rests


def arrival(pose: np.ndarray, s: int, g: int, p: RewardParams = RewardParams()) -> int:
    """The first step after s at which the reward counts the goal pose[g] as reached (reward.arrived,
    by straight-line distance, which is never longer than the map distance training uses, so this is a
    little early): where a training episode toward it would end. s itself if it starts there already."""
    for k in range(s, g + 1):
        d = float(np.hypot(*(pose[g, :2] - pose[k, :2])))
        if arrived(d, heading_error(pose[k, 2], pose[g, 2]), p):
            return k
    return g


def stop_pairs(steps: dict, ok: np.ndarray, lo: int, hi: int, window_s: float = 30.0,
               p: StopParams = StopParams(), rests: list = None) -> list:
    """(start, goal) step pairs within [lo, hi): starting from any rest (its last step), reaching a later
    intended stop (its first step) at most window_s later (the same window as goals.relabel's), through
    scored steps on one SLAM map. Every later intended stop in reach counts, so episodes pass through
    stops on the way to farther ones. A goal shorter than min_rest_s, or one the start has already
    reached (a short hop to about the same pose), is left out."""
    rate = float(steps['rate'])
    window = int(round(window_s * rate))
    shortest = max(1, int(round(p.min_rest_s * rate)))
    segment = map_segment(steps)
    pose = steps['pose'].astype(np.float64)
    rests = find_rests(steps, ok, p) if rests is None else rests
    rests = [r for r in rests if lo <= r.start and r.end < hi]
    goals = [r.start for r in rests if r.kind == 'intended' and r.end - r.start + 1 >= shortest]
    bad = np.cumsum(~ok)  # bad[k] - bad[j] counts unscored steps in (j, k]
    pairs = []
    for r in rests:
        s = r.end
        for g in goals:
            if s < g <= s + window and bad[g] == bad[s] and segment[g] == segment[s] and arrival(pose, s, g) > s:
                pairs.append((s, g))
    return pairs


def coverage(steps: dict, ok: np.ndarray, pairs: list) -> float:
    """The share of scored steps that some pair's episode runs through, each up to its first arrival as
    training's episodes end there."""
    n, pose = len(ok), steps['pose'].astype(np.float64)
    inside = np.zeros(n + 1, int)
    for s, g in pairs:
        inside[s] += 1
        inside[arrival(pose, s, g) + 1] -= 1
    return float(((np.cumsum(inside)[:n] > 0) & ok).sum() / max(ok.sum(), 1))


MOVING = 0.05  # m/s: for describing what followed a stop only (decides nothing)


def after_stop(steps: dict, rest: Rest, p: StopParams = StopParams(), look: int = 5) -> str:
    """What the robot did in the look steps after a rest: 'stays' (still at rest), 'turns' (turned
    without driving), 'reverses' (drove off the other way from how it arrived), 'drives on' (the same
    way) or 'sets off' (drove, having arrived without driving, e.g. from a turn in place). Driving means
    faster than MOVING."""
    v = np.nan_to_num(steps['odom_vel'])
    before = v[max(0, rest.start - look):rest.start, 0]
    later = v[rest.end + 1:rest.end + 1 + look]
    if not len(later) or ((np.abs(later[:, 0]) < p.rest_lin) & (np.abs(later[:, 1]) < p.rest_ang)).all():
        return 'stays'
    direction = lambda x: np.sign(x[np.argmax(np.abs(x))]) if len(x) and np.abs(x).max() > MOVING else 0
    went, goes = direction(before), direction(later[:, 0])
    if not goes:
        return 'turns'
    if not went:
        return 'sets off'
    return 'reverses' if went != goes else 'drives on'


def tally(paths: list, p: StopParams = StopParams(), window_s: float = 30.0) -> dict:
    """Counts over all sessions: rests by kind, intended stops' lengths and what followed, pairs, coverage."""
    out = {k: 0 for k in KINDS} | {'goals': 0, 'pairs': 0, 'covered': 0.0, 'scored': 0, 'lengths': {}, 'after': {}}
    for path in paths:
        steps = dict(np.load(path))
        ok = scored(steps)
        rests = find_rests(steps, ok, p)
        pairs = stop_pairs(steps, ok, 0, len(ok), window_s, p, rests)
        out['minutes'] = out.get('minutes', 0) + ok.sum() / float(steps['rate']) / 60
        for r in rests:
            out[r.kind] += 1
            if r.kind == 'intended':
                n = r.end - r.start + 1
                out['goals'] += n >= max(1, int(round(p.min_rest_s * float(steps['rate']))))
                band = '1' if n == 1 else '2' if n == 2 else '3-4' if n < 5 else '5-9' if n < 10 else '10+'
                out['lengths'][band] = out['lengths'].get(band, 0) + 1
                what = after_stop(steps, r, p)
                out['after'][what] = out['after'].get(what, 0) + 1
        out['pairs'] += len(pairs)
        out['covered'] += coverage(steps, ok, pairs) * ok.sum()
        out['scored'] += int(ok.sum())
    out['coverage'] = out['covered'] / max(out['scored'], 1)
    return out


def bump_timing(paths: list, p: StopParams = StopParams(), span_s: float = 2.0) -> tuple:
    """Bump onsets per 100 rests reached from motion, in each step before the rest, and the background
    rate per 100 steps: where collisions stop explaining stops."""
    counts, rests, onsets, steps_total = None, 0, 0, 0
    for path in paths:
        steps = dict(np.load(path))
        span = int(round(span_s * float(steps['rate'])))
        counts = np.zeros(span) if counts is None else counts
        ok = scored(steps)
        b = steps['bump'].any(axis=1)
        onset = b & ~np.r_[False, b[:-1]]
        onsets += onset.sum()
        steps_total += len(b)
        for r in find_rests(steps, ok, p):
            if r.kind != 'not from motion' and r.start > span:
                rests += 1
                counts += onset[r.start - span:r.start][::-1]
    return 100 * counts / max(rests, 1), 100 * onsets / max(steps_total, 1)


def main():
    import argparse
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', default=os.path.normpath(os.path.join(here, '..', '..', 'data', 'steps')))
    parser.add_argument('--sensitivity', action='store_true', help='also print the counts at other settings')
    args = parser.parse_args()
    paths = sorted(p for p in glob.glob(os.path.join(args.steps, '*.npz')) if not p.endswith('.maps.npz'))
    p = StopParams()

    print(f'rests in scored driving, by how the robot came to rest; pairs: from a rest to a later intended stop '
          f'within 30 s; covered: the share of scored driving some pair runs through, up to its arrival')
    print(f"{'session':16s} {'scored min':>10s} {'intended':>9s} {'bump':>5s} {'pushing':>8s} {'not from motion':>16s} {'pairs':>6s} {'covered':>8s}")
    for path in paths:
        t = tally([path], p)
        print(f"{os.path.basename(path)[:-4]:16s} {t['minutes']:10.1f} {t['intended']:9d} {t['bump']:5d} {t['pushing']:8d} "
              f"{t['not from motion']:16d} {t['pairs']:6d} {t['coverage']:8.0%}")
    t = tally(paths, p)
    print(f"{'all':16s} {t['minutes']:10.1f} {t['intended']:9d} {t['bump']:5d} {t['pushing']:8d} {t['not from motion']:16d} "
          f"{t['pairs']:6d} {t['coverage']:8.0%}")
    print('intended stops by length (0.1 s steps): ' + ', '.join(f"{k}: {t['lengths'].get(k, 0)}" for k in ('1', '2', '3-4', '5-9', '10+')))
    print('what followed them (the next 0.5 s): ' + ', '.join(f"{k}: {t['after'].get(k, 0)}" for k in ('stays', 'turns', 'sets off', 'drives on', 'reverses')))
    print('pairs and coverage by pair window: ' + ', '.join(
        f"{w} s: {x['pairs']} pairs, {x['coverage']:.0%}" for w, x in ((w, tally(paths, p, w)) for w in (30, 60, 120))))
    per_step, background = bump_timing(paths, p)
    print(f'bump onsets per 100 rests in each 0.1 s before the rest (background {background:.1f} per 100 steps): '
          + ' '.join(f'{x:.1f}' for x in per_step))
    if args.sensitivity:
        print('\nsensitivity (all sessions): intended (long enough to be goals) / bump / pushing / pairs / coverage')
        for field, values in (('rest_lin', (0.002, 0.005, 0.01, 0.02, 0.05)), ('rest_ang', (0.005, 0.01, 0.02, 0.05, 0.1)),
                              ('bump_before_s', (0.3, 0.6, 1.0, 1.2, 1.5)), ('min_rest_s', (0.1, 0.2, 0.5, 1.0))):
            for v in values:
                x = tally(paths, replace(p, **{field: v}))
                print(f"  {field} {v:<6g} {x['intended']:4d} ({x['goals']:3d}) / {x['bump']:3d} / {x['pushing']:3d} / {x['pairs']:5d} / {x['coverage']:.0%}"
                      + ('   (default)' if getattr(p, field) == v else ''))


if __name__ == '__main__':
    main()
