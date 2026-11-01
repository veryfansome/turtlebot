"""Train the driving model's world model on converted sessions and judge it on held-out driving.

    uv run python -m pet_learn.train_world [--steps ../data/steps] [--iters 3000] [--seed 0]

The last 20% of each session is held out. Each head is compared, at 0.1 s, 0.5 s and 1 s ahead, with a
naive baseline on the held-out windows: "the reading stays as it is" for sensors, "keep the current
velocity" for the pose change, and the base rate for bumps and slip. A ratio under 1 means the model
predicts better than the baseline. The model is saved to data/models/.
"""
import argparse
import glob
import os
import time

import numpy as np
import torch

from pet_learn.dataset import POSE_SCALE, TARGETS, Normalizer, SessionData, window_batch
from pet_learn.model import WorldModel, world_loss

HORIZONS = (1, 5, 10)


def to_torch(batch, device):
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


def baselines(batch: dict, steps_dt: float = 0.1):
    """Naive predictions for every target at every horizon, in the same (standardized) units."""
    out = {}
    for name, (size, binary) in TARGETS.items():
        if binary:
            continue
        out[name] = np.repeat(batch[name][:, :1], batch[name].shape[1], axis=1)  # stays as it is
    # Keep the current velocity: odom (forward, turn) scaled back to m/s and rad/s.
    k = np.arange(batch['pose'].shape[1])[None, :, None]
    v = batch['odom_raw'][:, :1]  # (B, 1, 2)
    pose = np.zeros_like(batch['pose'])
    pose[..., 0:1] = v[..., 0:1] * k * steps_dt / POSE_SCALE[0]
    pose[..., 2:3] = v[..., 1:2] * k * steps_dt / POSE_SCALE[2]
    out['pose'] = pose
    return out


def auc(scores, labels):
    """How well scores rank the positive labels above the negatives: 0.5 is chance, 1.0 is perfect.
    For rare events like bumps this is fairer than a calibration-sensitive loss."""
    pos, neg = labels > 0.5, labels <= 0.5
    if not pos.any() or not neg.any():
        return float('nan')
    ranks = np.argsort(np.argsort(scores)) + 1.0
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def evaluate(model, sessions, norm, device):
    """Model error / baseline error on the held-out windows, per head and horizon."""
    model.eval()
    rows = {}
    with torch.no_grad():
        batches = []
        for s in sessions:
            if len(s.val) == 0:
                continue
            b = window_batch(s, s.val, norm)
            b['odom_raw'] = s.targets['odom'][s.val[:, None] + np.arange(s.h + 1)[None, :]] * np.array([0.3, 1.5], np.float32)
            batches.append(b)
        batch = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
        t = to_torch({k: v for k, v in batch.items() if k != 'odom_raw'}, device)
        zs = model.rollout(model.encode(t['inputs'][:, 0]), t['actions'])
        preds = {k: v.cpu().numpy() for k, v in model.decode(zs, {n: t[n][:, :1] for n in TARGETS}).items()}
        preds['pose'] = np.concatenate([np.zeros_like(batch['pose'][:, :1]),
                                        model.pose(zs[:, :1], zs[:, 1:]).cpu().numpy()], axis=1)
        base = baselines(batch)
        for name, (size, binary) in list(TARGETS.items()) + [('pose', (3, False))]:
            target = batch[name]
            for h in HORIZONS:
                if binary:
                    model_err, base_err = auc(preds[name][:, h, 0], target[:, h, 0]), 0.5
                else:
                    model_err = float(((preds[name][:, h] - target[:, h]) ** 2).mean())
                    base_err = float(((base[name][:, h] - target[:, h]) ** 2).mean())
                rows[(name, h)] = (model_err, base_err)
    model.train()
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, '..', '..', 'data'))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', default=os.path.join(data_dir, 'steps'))
    parser.add_argument('--iters', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--z', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--eval-every', type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    paths = sorted(p for p in glob.glob(os.path.join(args.steps, '*.npz')) if not p.endswith('.maps.npz'))
    sessions = [SessionData(dict(np.load(p))) for p in paths]
    norm = Normalizer(sessions)
    train = [(i, s) for i, data in enumerate(sessions) for s in data.train]
    print(f'{len(sessions)} sessions, {len(train)} training windows, {sum(len(s.val) for s in sessions)} held out')

    model = WorldModel(sessions[0].inputs.shape[1], z_dim=args.z, hidden=args.hidden).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_state, best_it, started = float('inf'), None, 0, time.time()
    continuous = [n for n, (_, binary) in TARGETS.items() if not binary] + ['pose']
    for it in range(1, args.iters + 1):
        pick = rng.integers(0, len(train), args.batch)
        by_session = {}
        for j in pick:
            by_session.setdefault(train[j][0], []).append(train[j][1])
        # Half of each batch mirrored left to right.
        parts = [window_batch(sessions[i], np.array(st), norm, mirrored=bool(m))
                 for i, st in by_session.items() for m, st in enumerate(np.array_split(np.array(st), 2)) if len(st)]
        batch = to_torch({k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, args.device)
        loss, _ = world_loss(model, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if it % args.eval_every == 0:
            rows = evaluate(model, sessions, norm, args.device)
            # Held-out error, relative to the baselines, averaged over continuous heads and horizons.
            val = float(np.mean([rows[(n, h)][0] / max(rows[(n, h)][1], 1e-9) for n in continuous for h in HORIZONS]))
            if val < best:
                best, best_state, best_it = val, {k: v.clone() for k, v in model.state_dict().items()}, it
            if it % (args.eval_every * 5) == 0:
                print(f'iter {it}: train loss {loss.item():.3f}, held-out error / baseline {val:.3f}, '
                      f'best {best:.3f} at {best_it} ({time.time() - started:.0f} s)')

    model.load_state_dict(best_state)
    print(f'kept the model from iteration {best_it}')
    rows = evaluate(model, sessions, norm, args.device)
    print('\nheld-out: model error / baseline error (under 1 is better than the baseline);')
    print('bump and slip: AUC, how well it ranks the steps they happen at (0.5 chance, 1.0 perfect)')
    print(f'{"":10s}' + ''.join(f'{h / 10:>8.1f} s' for h in HORIZONS))
    for name in list(TARGETS) + ['pose']:
        binary = name in TARGETS and TARGETS[name][1]
        cell = (lambda m, b: m) if binary else (lambda m, b: m / max(b, 1e-9))
        print(f'{name:10s}' + ''.join(f'{cell(*rows[(name, h)]):>10.2f}' for h in HORIZONS))
    os.makedirs(os.path.join(data_dir, 'models'), exist_ok=True)
    out = os.path.join(data_dir, 'models', f'world-{time.strftime("%Y%m%d-%H%M%S")}.pt')
    torch.save({'model': model.state_dict(), 'mean': norm.mean, 'std': norm.std,
                'input_dim': sessions[0].inputs.shape[1], 'z': args.z, 'hidden': args.hidden}, out)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
