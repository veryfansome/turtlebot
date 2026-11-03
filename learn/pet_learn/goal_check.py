"""Does the driver's policy use its goal? Checked offline, on held-out driving.

    uv run python -m pet_learn.goal_check --driver ../data/models/driver-<time>.pt

Each held-out state is paired with a goal the recorded driving went on to reach, and the policy's
command is computed twice: with the goal, and with the goal mirrored left to right (its goal features
flipped as train_driver's GOAL_MIRROR does). The Driver is symmetric by construction, so the policy
with the goal mirrored is exactly the policy for the mirror image of the whole scene: a situation as
real as the original, not an unusual input. Its turn splits into two parts:

- the goal's part, (turn - mirrored turn) / 2, signed so that positive turns toward the goal's side
  (its bearing; within the reward's arrival radius, where only heading is left to fix, the side of its
  heading error): how much the goal steers the policy;
- the state's part, (turn + mirrored turn) / 2: what it does whatever the goal's side, for example
  carrying on with the previous command, which is an input (dataset.inputs).

Reported per group, with 90% intervals from resampling 30 s blocks of held-out driving (goals overlap,
so resampling goals would treat neighbours as independent; the intervals cover the driving's noise,
not training's: compare drivers by several seeds each): the goal's part (rad/s), the state's part
(median size, rad/s), and the share of states where the goal sets the turn's direction (mirroring
the goal reverses it, which happens exactly when the goal's part outweighs the state's). Groups, for
states farther from the goal than the reward's arrival radius: what the robot was doing (stopped: at
rest, stops.at_rest; turning: the previous command turned, as the policy sees it; straight: neither)
and the goal's bearing in 30-degree bands, which only group the results; within it, where only the
heading is left to fix, a row of its own. And the approach: the policy's forward command by
distance to the goal, beside the recorded command (does it slow down as the person did?).

Pairs end at their first arrival (the reward's test: within 0.2 m and 20 degrees, here by straight-line
distance, which is never longer than the map distance training uses, so it ends them a little early),
as training's episodes do. States after it, where the robot has arrived and the recorded driving sits
or moves on, are never trained on; they're reported on a row of their own.

Goals come in two families: windows, sampled as train_driver's default episodes (a random scored step,
with up to 30 s of scored driving before it), and stops (from a rest to a later intended stop within 30 s,
stops.stop_pairs). By default the region is the last 20% of each session: the driver never trains on
it, and its first half only picked the world model's iteration, by prediction error. The last 10%
alone (--region test) holds few stop pairs. This is a check, not a ranking: it can't say whether the
robot would reach the goal; that needs closed-loop driving.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from pet_learn.dataset import FORWARD_CAP, TURN_CAP, SessionData
from pet_learn.episodes import map_segment, scored
from pet_learn.goals import goal_features
from pet_learn.reward import RewardParams, arrived
from pet_learn.stops import at_rest, stop_pairs
from pet_learn.train_driver import GOAL_MIRROR, REVERSE_CAP, Driver, encode_pair, input_mirror, world_model
from pet_learn.train_world import default_device

CAPS = np.array([FORWARD_CAP, TURN_CAP])
BANDS = np.arange(0, 181, 30)
DISTANCES = [0, 0.25, 0.5, 1, 2, np.inf]


def window_pairs(steps: dict, ok: np.ndarray, lo: int, hi: int, n: int, rng, window_s: float = 30.0) -> list:
    """(start, goal) pairs sampled as train_driver's default episodes, windows (goals.relabel): a random scored step
    in [lo, hi), with the scored driving on one SLAM map up to window_s before it."""
    rate = float(steps['rate'])
    window = int(round(window_s * rate))
    segment = map_segment(steps)
    candidates = np.flatnonzero(ok[lo:hi]) + lo
    pairs = []
    for g in (rng.choice(candidates, n) if len(candidates) else []):
        s = g
        while s > max(lo, g - window) and ok[s - 1] and segment[s - 1] == segment[g]:
            s -= 1
        if g - s >= 2:
            pairs.append((int(s), int(g)))
    return pairs


def pair_states(steps: dict, pairs: list, block_s: float = 30.0) -> dict:
    """Every step from each pair's start up to (not including) its goal, with the goal's features, which
    side it's on (+1 left, -1 right, 0 dead ahead or behind), its bearing (deg), distance (m), what the
    robot was doing (0 stopped, 1 straight, 2 turning), whether it's at or after the pair's first arrival
    (training never sees those), which pair (goal) it belongs to, and which block of block_s of driving
    its goal falls in (for intervals: neighbouring goals share their driving)."""
    pose = steps['pose'].astype(np.float64)
    rest = at_rest(steps)
    turned = np.r_[False, steps['cmd'][:-1, 1] != 0]  # the previous command, as the policy sees it
    rp = RewardParams()
    radius = rp.goal_radius
    block = int(round(block_s * float(steps['rate'])))
    out = {k: [] for k in ('step', 'goal', 'side', 'bearing', 'distance', 'doing', 'after', 'pair', 'block')}
    for i, (s, g) in enumerate(pairs):
        ks = np.arange(s, g)
        f = goal_features(pose[ks], pose[g])
        dist = np.hypot(pose[g, 0] - pose[ks, 0], pose[g, 1] - pose[ks, 1])
        side = np.where(dist > radius, np.sign(f[:, 1]), np.sign(f[:, 3]))
        out['step'].append(ks)
        out['goal'].append(f)
        out['side'].append(side)
        out['bearing'].append(np.degrees(np.abs(np.arctan2(f[:, 1], f[:, 0]))))
        out['distance'].append(dist)
        here = np.array([arrived(d, h, rp) for d, h in zip(dist, np.arctan2(f[:, 3], f[:, 4]))])
        out['after'].append(np.maximum.accumulate(np.r_[False, here[1:]]))  # as relabel: arrival from the step after the start
        out['doing'].append(np.where(rest[ks], 0, np.where(turned[ks], 2, 1)))
        out['pair'].append(np.full(len(ks), i))
        out['block'].append(np.full(len(ks), g // block))
    if not pairs:
        return None
    return {k: np.concatenate(v) for k, v in out.items()}


def goal_parts(turn: np.ndarray, turn_mirrored: np.ndarray, side: np.ndarray):
    """The goal's part of the turn (signed toward the goal's side), the state's part, and whether
    mirroring the goal reversed the turn."""
    return side * (turn - turn_mirrored) / 2, (turn + turn_mirrored) / 2, np.sign(turn) != np.sign(turn_mirrored)


def interval(values: np.ndarray, cluster: np.ndarray, rng, n: int = 1000) -> tuple:
    """Mean and a 90% interval from resampling clusters (blocks of driving) with replacement."""
    if not len(values):
        return float('nan'), float('nan'), float('nan')
    ids, inverse = np.unique(cluster, return_inverse=True)
    sums, counts = np.bincount(inverse, values, len(ids)), np.bincount(inverse, minlength=len(ids))
    draws = rng.integers(0, len(ids), (n, len(ids)))
    means = sums[draws].sum(1) / np.maximum(counts[draws].sum(1), 1)
    return float(values.mean()), float(np.percentile(means, 5)), float(np.percentile(means, 95))


def check(states: dict, policy: np.ndarray, mirrored: np.ndarray, recorded: np.ndarray, rng) -> dict:
    """The tables for one goal family: policy and mirrored are commands (n, 2) in m/s, rad/s."""
    goal, state, flips = goal_parts(policy[:, 1], mirrored[:, 1], states['side'])
    pair = states['block']
    before = ~states['after']
    far = before & (states['distance'] > RewardParams().goal_radius)

    def row(m):
        return {'states': int(m.sum()), 'goal_turn': interval(goal[m], pair[m], rng),
                'state_turn': float(np.median(np.abs(state[m]))) if m.any() else float('nan'),
                'flips': interval(flips[m].astype(float), pair[m], rng)}

    out = {'states': int(len(goal)), 'goals': int(len(np.unique(states['pair']))), 'blocks': int(len(np.unique(pair))),
           'far': row(far), 'at_goal': row(before & ~far), 'after': row(states['after'])}
    for code, name in enumerate(('stopped', 'straight', 'turning')):
        out[name] = row(far & (states['doing'] == code))
    out['bearing'] = {}
    for a, b in zip(BANDS[:-1], BANDS[1:]):
        m = (states['bearing'] >= a) & (states['bearing'] < b if b < 180 else states['bearing'] <= b)
        out['bearing'][f'{a}-{b}'] = row(far & m)
    out['approach'] = {}
    for a, b in zip(DISTANCES[:-1], DISTANCES[1:]):
        m = before & (states['distance'] >= a) & (states['distance'] < b)
        out['approach'][f'{a}-{b}'] = {'states': int(m.sum()),
                                       'policy_forward': float(np.median(policy[m, 0])) if m.any() else float('nan'),
                                       'recorded_forward': float(np.median(recorded[m, 0])) if m.any() else float('nan')}
    m = states['after']
    out['approach']['after arriving'] = {
        'states': int(m.sum()), 'policy_forward': float(np.median(policy[m, 0])) if m.any() else float('nan'),
        'recorded_forward': float(np.median(recorded[m, 0])) if m.any() else float('nan')}
    return out


def load_driver(path: str, device: str):
    ckpt = torch.load(path, weights_only=False, map_location='cpu')
    world = world_model(ckpt)
    world.load_state_dict(ckpt['world'])
    driver = Driver(ckpt['z'])
    driver.load_state_dict(ckpt['driver'])
    return world.to(device).eval(), driver.to(device).eval(), ckpt


def report(name: str, r: dict) -> None:
    cell = lambda t, fmt: f'{fmt.format(t[0])} [{fmt.format(t[1])}, {fmt.format(t[2])}]'
    print(f"\n{name}: {r['goals']} goals in {r['blocks']} blocks of 30 s, {r['states']} states")
    print(f"  {'':30s} {'states':>7s}  {'goal part of the turn, rad/s':>30s}  {'state part':>12s}  {'goal sets direction':>20s}")
    radius = RewardParams().goal_radius
    rows = [(f'farther than {radius:g} m', r['far'])] + [(f'  {k}', r[k]) for k in ('stopped', 'straight', 'turning')]
    rows += [(f'  bearing {k} deg', v) for k, v in r['bearing'].items()]
    rows += [(f'within {radius:g} m (heading)', r['at_goal']), ('after arriving (never trained)', r['after'])]
    for label, v in rows:
        print(f"  {label:30s} {v['states']:7d}  {cell(v['goal_turn'], '{:+.2f}'):>30s}  {v['state_turn']:12.2f}  "
              f"{cell(v['flips'], '{:.0%}'):>20s}")
    print(f"  approach, median forward command (m/s) by distance to the goal (m):")
    for k, v in r['approach'].items():
        label = k if k == 'after arriving' else f'{k} m, before arriving'
        print(f"    {label:24s} {v['states']:6d} states  policy {v['policy_forward']:+.2f}  recorded {v['recorded_forward']:+.2f}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, '..', '..', 'data'))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--driver', required=True, help='a driver checkpoint from pet_learn.train_driver')
    parser.add_argument('--steps', default=os.path.join(data_dir, 'steps'))
    parser.add_argument('--region', choices=('heldout', 'test'), default='heldout',
                        help='heldout: the last 20%% of each session; test: the last 10%%')
    parser.add_argument('--goals', type=int, default=200, help='window episodes per session')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--json', help='also write the results here')
    parser.add_argument('--device', default=default_device())
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    world, driver, ckpt = load_driver(args.driver, args.device)
    mirror_x = input_mirror(ckpt['input_dim']).to(args.device)
    mirror_g = GOAL_MIRROR.numpy()
    collected = {'stops': [], 'windows': []}
    for path in sorted(p for p in glob.glob(os.path.join(args.steps, '*.npz')) if not p.endswith('.maps.npz')):
        steps = dict(np.load(path))
        data = SessionData(steps)
        if data.inputs.shape[1] != ckpt['input_dim']:
            raise SystemExit(f"{args.driver} takes {ckpt['input_dim']} inputs per step, the data has {data.inputs.shape[1]}")
        ok = scored(steps)
        lo, hi = (data.split if args.region == 'heldout' else data.test_split), len(ok)
        families = {'stops': stop_pairs(steps, ok, lo, hi), 'windows': window_pairs(steps, ok, lo, hi, args.goals, rng)}
        for family, pairs in families.items():
            st = pair_states(steps, pairs)
            if st is None:
                continue
            with torch.no_grad():
                s = encode_pair(world, torch.as_tensor(data.inputs[st['step']], device=args.device), mirror_x)
                g = torch.as_tensor(st['goal'], device=args.device)
                policy = driver.policy(s, g)[0].cpu().numpy() * CAPS
                mirrored = driver.policy(s, g * torch.as_tensor(mirror_g, device=args.device))[0].cpu().numpy() * CAPS
            recorded = np.clip(steps['cmd'][st['step']], [-REVERSE_CAP * CAPS[0], -CAPS[1]], CAPS)  # as the policy learns them
            collected[family].append((st, policy, mirrored, recorded, os.path.basename(path)))

    results = {'driver': args.driver, 'region': args.region}
    print(f'{args.driver}, {args.region} region (90% intervals from resampling 30 s blocks of held-out driving)')
    for family in ('stops', 'windows'):
        parts = collected[family]
        if not parts:
            print(f'\n{family}: no pairs')
            continue
        offset, block, merged = 0, 0, []
        for st, *_ in parts:  # pair and block ids unique across sessions
            merged.append(dict(st, pair=st['pair'] + offset, block=st['block'] + block))
            offset, block = offset + st['pair'].max() + 1, block + st['block'].max() + 1
        st = {k: np.concatenate([m[k] for m in merged]) for k in merged[0]}
        policy, mirrored, recorded = (np.concatenate([p[i] for p in parts]) for i in (1, 2, 3))
        results[family] = check(st, policy, mirrored, recorded, rng)
        report(f'{family} goals', results[family])
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
