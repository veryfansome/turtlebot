"""How far ahead does the world model stay believable? An open-loop replay on held-out driving.

    uv run python -m pet_learn.replay_check --world ../data/models/world-<time>.pt

Closed-loop rollouts (a policy driving inside the world model, for a driver's fitness) chain the
model's own predictions for up to 30 s, but it's trained only up to 1 s ahead (dataset.H) and scored up to 2 s
(dataset.EVAL_H).
This measures how its predictions degrade beyond that, with the recorded commands instead of a policy's:
from a held-out step, it encodes the inputs once, then steps the dynamics on the commands the person
actually sent, never seeing another reading, and compares what it predicts with what happened.

At each horizon (0.1 s to 30 s), only starts whose driving is usable all the way there (and on one SLAM
map) count, so long horizons have fewer. Reported:

- each continuous reading's error relative to "it stays as it is" (as train_world scores them, in
  standardized units; under 1 beats it), and bump AUC (0.5 is chance);
- the pose: where the robot ends up, in the frame of the start, against the SLAM pose, three ways:
  direct, the pose head on (start state, state k), as trained (up to 1 s); chained per step, the pose
  head on each step (state j, state j+1), composed; chained per second, on 1 s chunks. Against "keep the
  current velocity" (odometry at the start, for the whole horizon). Errors in m and deg, medians.

The commands are the recorded ones, so this says nothing about what the model predicts for commands
the person never gave (a policy's): it's the best case for closed-loop rollouts, not their fitness.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from pet_learn.dataset import MASKS, POSE_SCALE, TARGETS, Normalizer, SessionData
from pet_learn.episodes import map_segment, scored, usable
from pet_learn.model import current_readings
from pet_learn.train_driver import world_model
from pet_learn.train_world import auc, default_device

HORIZONS = (1, 5, 10, 20, 50, 100, 200, 300)  # steps ahead (0.1 s each)


def load_world(path: str, device: str):
    ckpt = torch.load(path, weights_only=False, map_location='cpu')
    model = world_model(ckpt)
    model.load_state_dict(ckpt['model'])
    norm = Normalizer.__new__(Normalizer)  # the training data's standardization, as saved with the model
    norm.mean, norm.std = ckpt['mean'], ckpt['std']
    return model.to(device).eval(), norm, ckpt


def replay_starts(steps: dict, lo: int, stride: int) -> tuple:
    """Starts from lo on, every stride steps, at usable steps, and for each how many steps ahead its
    driving stays usable, scored-or-not aside, and on the start's SLAM map (capped at max(HORIZONS)).
    Driving after the last scored step (the session's unscored tail) doesn't count."""
    ok = usable(steps)
    u = np.flatnonzero(scored(steps))
    end = int(u[-1]) + 1 if len(u) else 0
    segment = map_segment(steps)
    starts = np.arange(lo, end, stride)
    starts = starts[ok[starts]]
    reach = np.zeros(len(starts), int)
    for i, s in enumerate(starts):
        k = 0
        while k < max(HORIZONS) and s + k + 1 < end and ok[s + k + 1] and segment[s + k + 1] == segment[s]:
            k += 1
        reach[i] = k
    return starts, reach


def compose(deltas: np.ndarray) -> np.ndarray:
    """Pose changes (B, k, 3) in m, m, rad, each in the frame where the previous one ended, chained into
    the pose after each (B, k, 3), in the frame of the first."""
    out = np.zeros_like(deltas)
    x = y = th = np.zeros(deltas.shape[0])
    for j in range(deltas.shape[1]):
        dx, dy, dth = deltas[:, j, 0], deltas[:, j, 1], deltas[:, j, 2]
        x, y = x + np.cos(th) * dx - np.sin(th) * dy, y + np.sin(th) * dx + np.cos(th) * dy
        th = th + dth
        out[:, j] = np.stack([x, y, th], axis=1)
    return out


def true_pose(pose: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The SLAM pose at idx (B, k) in the frame of the pose at idx[:, 0], in m, m, rad (heading wrapped)."""
    p = pose[idx]
    c, s = np.cos(p[:, :1, 2]), np.sin(p[:, :1, 2])
    dx, dy = p[..., 0] - p[:, :1, 0], p[..., 1] - p[:, :1, 1]
    dth = np.arctan2(np.sin(p[..., 2] - p[:, :1, 2]), np.cos(p[..., 2] - p[:, :1, 2]))
    return np.stack([c * dx + s * dy, -s * dx + c * dy, dth], axis=-1)


def replay(model, norm, data: SessionData, starts: np.ndarray, device: str) -> dict:
    """The model's predictions along max(HORIZONS) steps of recorded commands from each start, and the truth."""
    n, h = len(data.inputs), max(HORIZONS)
    idx = np.minimum(starts[:, None] + np.arange(h + 1)[None, :], n - 1)  # past the end: clipped, never counted
    batch = norm({name: data.targets[name][idx].astype(np.float32) for name in TARGETS})
    masks = {key: m[idx] for key, m in data.masks.items()}
    scale = np.array(POSE_SCALE)
    with torch.no_grad():
        t = lambda a: torch.as_tensor(a, device=device)
        current = current_readings({**{k: t(v) for k, v in batch.items()},
                                    **{f'{name}_seen': t(masks[key]) for name, key in MASKS.items()}})
        zs = model.rollout(model.encode(t(data.inputs[idx[:, 0]])), t(data.actions[idx[:, :-1]]))
        preds = {k: v.cpu().numpy() for k, v in model.decode(zs, current).items()}
        direct = model.pose(zs[:, :1], zs[:, 1:]).cpu().numpy() * scale
        per_step = torch.stack([model.pose(zs[:, j:j + 1], zs[:, j + 1:j + 2])[:, 0] for j in range(h)], 1).cpu().numpy() * scale
        per_second = torch.stack([model.pose(zs[:, j:j + 1], zs[:, j + 10:j + 11])[:, 0] for j in range(0, h, 10)], 1).cpu().numpy() * scale
    odom = data.targets['odom'][idx[:, 0]] * np.array([0.3, 1.5])  # m/s, rad/s at the start (the target is divided by the caps)
    return {'batch': batch, 'masks': masks, 'preds': preds, 'idx': idx,
            'pose': {'direct': direct, 'chained per step': compose(per_step),
                     'chained per second': np.repeat(compose(per_second), 10, axis=1)},
            'truth': true_pose(data.pose, idx)[:, 1:], 'odom': odom}


def summarize(runs: list) -> dict:
    """Per horizon: each reading's error relative to "it stays as it is", bump AUC, and the pose errors."""
    out = {}
    for h in HORIZONS:
        row = {'starts': 0}
        keep = [r['reach'] >= h for r in runs]
        row['starts'] = int(sum(k.sum() for k in keep))
        if row['starts'] == 0:
            out[f'{h / 10:g}s'] = row
            continue
        for name, (size, binary) in TARGETS.items():
            if binary:
                if name == 'bump':
                    p = np.concatenate([r['preds'][name][k, h, 0] for r, k in zip(runs, keep)])
                    y = np.concatenate([r['batch'][name][k, h, 0] for r, k in zip(runs, keep)])
                    row['bump_auc'] = auc(p, y)
                continue
            err = base = 0.0
            for r, k in zip(runs, keep):
                m = k.copy()
                if name in MASKS:  # real now and h steps ahead
                    m &= (r['masks'][MASKS[name]][:, 0, 0] > 0.5) & (r['masks'][MASKS[name]][:, h, 0] > 0.5)
                target = r['batch'][name][m, h]
                err += float(((r['preds'][name][m, h] - target) ** 2).sum())
                base += float(((r['batch'][name][m, 0] - target) ** 2).sum())
            row[name] = err / base if base > 0 else float('nan')
        keep = [k & (r['masks']['odom_ok'][:, 0, 0] > 0.5) for r, k in zip(runs, keep)]  # the velocity baseline needs odometry
        truth = np.concatenate([r['truth'][k, h - 1] for r, k in zip(runs, keep)])
        odom = np.concatenate([r['odom'][k] for r, k in zip(runs, keep)])
        t = h / 10
        th = odom[:, 1] * t  # keep the current velocity: an arc
        arc = np.where(np.abs(odom[:, 1]) > 1e-6, odom[:, 0] / np.where(np.abs(odom[:, 1]) > 1e-6, odom[:, 1], 1), 0)
        straight = np.abs(odom[:, 1]) <= 1e-6
        bx = np.where(straight, odom[:, 0] * t, arc * np.sin(th))
        by = np.where(straight, 0.0, arc * (1 - np.cos(th)))
        guesses = {'keep velocity': np.stack([bx, by, th], axis=1)}
        for way in ('direct', 'chained per step', 'chained per second'):
            if way == 'chained per second' and h % 10:  # whole seconds only
                continue
            guesses[way] = np.concatenate([r['pose'][way][k, h - 1] for r, k in zip(runs, keep)])
        row['moved_m'] = float(np.median(np.hypot(truth[:, 0], truth[:, 1])))
        for way, g in guesses.items():
            dist = np.hypot(g[:, 0] - truth[:, 0], g[:, 1] - truth[:, 1])
            ang = np.degrees(np.abs(np.arctan2(np.sin(g[:, 2] - truth[:, 2]), np.cos(g[:, 2] - truth[:, 2]))))
            row[f'pose {way}'] = {'m': float(np.median(dist)), 'deg': float(np.median(ang))}
        out[f'{t:g}s'] = row
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, '..', '..', 'data'))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--world', required=True, help='a world model checkpoint from pet_learn.train_world')
    parser.add_argument('--steps', default=os.path.join(data_dir, 'steps'))
    parser.add_argument('--region', choices=('heldout', 'test'), default='heldout',
                        help='heldout: from 80%% of each session\'s scored driving on; test: from 90%%')
    parser.add_argument('--stride', type=int, default=10, help='steps between replay starts')
    parser.add_argument('--json', help='also write the results here')
    parser.add_argument('--device', default=default_device())
    args = parser.parse_args()

    model, norm, ckpt = load_world(args.world, args.device)
    runs = []
    for path in sorted(p for p in glob.glob(os.path.join(args.steps, '*.npz')) if not p.endswith('.maps.npz')):
        steps = dict(np.load(path))
        data = SessionData(steps)
        if data.inputs.shape[1] != ckpt['input_dim']:
            raise SystemExit(f"{args.world} takes {ckpt['input_dim']} inputs per step, the data has {data.inputs.shape[1]}")
        starts, reach = replay_starts(steps, data.split if args.region == 'heldout' else data.test_split, args.stride)
        if len(starts):
            runs.append(dict(replay(model, norm, data, starts, args.device), reach=reach))
    results = summarize(runs)

    print(f'{args.world}: open-loop replay of recorded commands, {args.region} region, a start every {args.stride / 10:g} s')
    heads = [n for n, (_, b) in TARGETS.items() if not b]
    print('\nreadings: error relative to "it stays as it is" (under 1 beats it); bump: AUC')
    print(f"{'ahead':>6} {'starts':>6} " + ' '.join(f'{n:>8}' for n in heads) + f" {'bump':>6}")
    for k, row in results.items():
        if row['starts'] == 0:
            print(f'{k:>6} {0:6d}  (no driving usable that far)')
            continue
        print(f"{k:>6} {row['starts']:6d} " + ' '.join(f'{row[n]:8.2f}' for n in heads) + f" {row.get('bump_auc', float('nan')):6.2f}")
    ways = ('keep velocity', 'direct', 'chained per step', 'chained per second')
    print('\npose: median error, m / deg (moved: median distance from the start, SLAM)')
    print(f"{'ahead':>6} {'moved':>6} " + ' '.join(f'{w:>19}' for w in ways))
    for k, row in results.items():
        if row['starts']:
            print(f"{k:>6} {row['moved_m']:6.2f} " + ' '.join(f"{row[f'pose {w}']['m']:10.2f} / {row[f'pose {w}']['deg']:5.0f}" if f'pose {w}' in row else f"{'-':>19}" for w in ways))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'world': args.world, 'region': args.region, 'stride': args.stride, 'results': results}, f, indent=1)


if __name__ == '__main__':
    main()
