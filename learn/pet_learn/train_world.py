"""Train the driving model's world model on converted sessions and judge it on held-out driving.

    uv run python -m pet_learn.train_world [--steps ../data/steps] [--patience 2000] [--seed 0] [--genome G.json]

The last 20% of each session's scored driving is held out: the first half of it (val) picks the best model and when to
stop, and the second half (test) is what's reported, so the choice can't flatter the result. Each
head is compared, at 0.1 s, 0.5 s, 1 s and 2 s ahead (training reaches 1 s, dataset.H; held-out windows 2 s,
dataset.EVAL_H), with a
naive baseline on the held-out windows: "the reading stays as it is" for sensors, "keep the current
velocity" for the pose change, and the base rate for bumps and slip. A ratio under 1 means the model
predicts better than the baseline. Training stops once the val error hasn't improved for
--patience iterations (or at --iters), and keeps the best model, saved to data/models/ (or --out-dir).

What's trained, and how, is a genome (pet_learn.genome): the architecture, the objective, the optimizer
and the batches. Without --genome it's the baseline, with --batch, --lr and the rest setting its params.
With --final-steps, the model is also scored on every window of those sessions (recorded after the
search began, so never optimized against): the evolve search's final split.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

from pet_learn.dataset import MASKS, POSE_SCALE, TARGETS, Normalizer, SessionData, window_batch
from pet_learn.model import current_readings
from pet_learn import genome as genomes, runs

HORIZONS = (1, 5, 10, 20)  # steps ahead scored: 0.1, 0.5, 1 and 2 s (training reaches 1 s, dataset.H)


def default_device() -> str:
    """A GPU when there is one: CUDA (a RunPod pod), or the Mac's (MPS: about 4x faster than its CPU
    at batch 1024, 1.6x at 256)."""
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def score(rows) -> float:
    """One number for a model: its error relative to the naive baselines, over the continuous heads
    (and the pose change) and the horizons, combined as a geometric mean. Lower is better; under 1 beats
    them. Geometric, so halving any head's error counts the same whatever its scale: an arithmetic mean
    was dominated by the pose change 0.1 s ahead, whose near-perfect baseline makes its ratio 2 for a
    trained model and 25-67 early on, against 0.06-1.3 for the rest (2026-11-01). Ratios that can't be
    computed (a reading never recorded in the held-out driving) are left out."""
    continuous = [n for n, (_, binary) in TARGETS.items() if not binary] + ['pose']
    ratios = np.array([rows[(n, h)][0] / max(rows[(n, h)][1], 1e-9) for n in continuous for h in HORIZONS], dtype=float)
    return float(np.exp(np.nanmean(np.log(np.maximum(ratios, 1e-9)))))


def by_horizon(rows) -> dict:
    """{'<h>s': {head: ratio to its baseline (AUC for bump and slip)}}, as run records keep them."""
    return {f'{h / 10:g}s': {n: (rows[(n, h)][0] if n in TARGETS and TARGETS[n][1] else rows[(n, h)][0] / max(rows[(n, h)][1], 1e-9))
                             for n in list(TARGETS) + ['pose']} for h in HORIZONS}


def unused_path(directory: str, prefix: str) -> str:
    """<directory>/<prefix>-<time>.pt, numbered if a model saved in the same second has it."""
    base = os.path.join(directory, f'{prefix}-{time.strftime("%Y%m%d-%H%M%S")}')
    out, k = base + '.pt', 1
    while os.path.exists(out):
        k += 1
        out = f'{base}-{k}.pt'
    return out


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


def evaluate(model, sessions, norm, device, part='val'):
    """Model error / baseline error on held-out windows (SessionData's val or test), per head and horizon."""
    model.eval()
    rows = {}
    with torch.no_grad():
        batches = []
        for s in sessions:
            windows = getattr(s, part)
            if len(windows) == 0:
                continue
            b = window_batch(s, windows, norm, h=s.eval_h)
            b['odom_raw'] = s.targets['odom'][windows[:, None] + np.arange(s.eval_h + 1)[None, :]] * np.array([0.3, 1.5], np.float32)
            batches.append(b)
        if not batches:
            raise ValueError(f'no {part} windows in any session')
        batch = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
        t = to_torch({k: v for k, v in batch.items() if k != 'odom_raw'}, device)
        zs = model.rollout(model.encode(t['inputs'][:, 0]), t['actions'])
        preds = {k: v.cpu().numpy() for k, v in model.decode(zs, current_readings(t)).items()}
        preds['pose'] = np.concatenate([np.zeros_like(batch['pose'][:, :1]),
                                        model.pose(zs[:, :1], zs[:, 1:]).cpu().numpy()], axis=1)
        base = baselines(batch)
        for name, (size, binary) in list(TARGETS.items()) + [('pose', (3, False))]:
            target = batch[name]
            for h in HORIZONS:
                # Where the reading is real, now and h steps ahead (the baseline needs it now).
                if name in MASKS:
                    keep = (batch[MASKS[name]][:, h, 0] > 0.5) & (batch[MASKS[name]][:, 0, 0] > 0.5)
                elif name == 'pose':  # its baseline keeps the odometry's current velocity
                    keep = batch['odom_ok'][:, 0, 0] > 0.5
                else:
                    keep = slice(None)
                if binary:
                    model_err, base_err = auc(preds[name][:, h, 0], target[:, h, 0]), 0.5
                elif not np.any(keep):  # not recorded in the held-out driving
                    model_err, base_err = float('nan'), float('nan')
                else:
                    model_err = float(((preds[name][keep, h] - target[keep, h]) ** 2).mean())
                    base_err = float(((base[name][keep, h] - target[keep, h]) ** 2).mean())
                rows[(name, h)] = (model_err, base_err)
    model.train()
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, '..', '..', 'data'))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', default=os.path.join(data_dir, 'steps'))
    parser.add_argument('--iters', type=int, default=50000, help='at most this many iterations')
    parser.add_argument('--patience', type=int, default=2000,
                        help='stop after this many iterations without a better val error')
    parser.add_argument('--genome', help='a genome JSON (pet_learn.genome); default: the baseline')
    parser.add_argument('--out-dir', default=os.path.join(data_dir, 'models'))
    parser.add_argument('--final-steps', help='also score on every window of the sessions in this folder')
    # Override some of the baseline genome's params (genome.BASELINE: batch 2048 at 6e-4, hide 0.1,
    # dropout 0.1 in the encoder and dynamics, weight decay 1e-3, loss weights lidar 3, IR 2, bump 0.5; z 64 and hidden 128 from the impl); for a plain run only, not with
    # --genome. The rest (depth, dropout, consistency, warmup, decay) are set by a genome JSON: see README.md.
    parser.add_argument('--hide', type=float, help='chance each sensor is hidden from the encoder in a training window (dataset.hide)')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--hidden', type=int)
    parser.add_argument('--z', type=int)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=default_device())
    parser.add_argument('--eval-every', type=int, default=100)
    args = parser.parse_args()
    flags = {'arch': {'z': args.z, 'hidden': args.hidden}, 'optim': {'lr': args.lr, 'weight_decay': args.weight_decay, 'batch': args.batch},
             'batcher': {'hide': args.hide}}
    overrides = {axis: {k: v for k, v in params.items() if v is not None} for axis, params in flags.items()}
    if args.genome and any(overrides.values()):
        sys.exit("a genome sets its own params: don't combine --genome with --batch, --lr, --hide, --hidden, --z or --weight-decay")
    genome = genomes.load(args.genome, overrides)
    record = runs.new_record('world', parser, args)
    record['genome'] = genome
    print(f'genome: {genomes.recipe(genome)}')

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    paths = sorted(p for p in glob.glob(os.path.join(args.steps, '*.npz')) if not p.endswith('.maps.npz'))
    sessions = [SessionData(dict(np.load(p))) for p in paths]
    norm = Normalizer(sessions)
    train = [(i, s) for i, data in enumerate(sessions) for s in data.train]
    print(f'{len(sessions)} sessions, {len(train)} training windows, {sum(len(s.val) for s in sessions)} val '
          f'(picks the model), {sum(len(s.test) for s in sessions)} test (reported)')
    for p, s in zip(paths, sessions):
        print(f'  {os.path.basename(p)[:-4]}: {len(s.train)} / {len(s.val)} / {len(s.test)}')
    if not all(sum(len(getattr(s, part)) for s in sessions) for part in ('train', 'val', 'test')):
        sys.exit('need training, val and test windows: too little usable driving')
    runs.add_sessions(record, paths, sessions)
    record['windows'] = {part: sum(len(getattr(s, part)) for s in sessions) for part in ('train', 'val', 'test')}

    arch, arch_params = genomes.impl(genome, 'arch')
    objective, objective_params = genomes.impl(genome, 'objective')
    optim, optim_params = genomes.impl(genome, 'optim')
    batcher, batcher_params = genomes.impl(genome, 'batcher')
    model = arch.build(sessions[0].inputs.shape[1], **arch_params).to(args.device)
    opt, scheduler, batch_size = optim.make(model.parameters(), **optim_params)
    next_batch = batcher.make(sessions, train, norm, rng, batch_size, **batcher_params)
    best, best_state, best_it, started, stopped = float('inf'), None, 0, time.time(), None
    for it in range(1, args.iters + 1):
        batch = to_torch(next_batch(), args.device)
        loss, _ = objective.loss(model, batch, **objective_params)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        if it % args.eval_every == 0:
            if not torch.isfinite(loss):  # the weights are gone; the best model so far (before it) is kept
                stopped = f'non-finite training loss at iteration {it}'
                print(f'stopped: {stopped}')
                break
            val = score(evaluate(model, sessions, norm, args.device))
            if np.isfinite(val) and val < best:
                best, best_state, best_it = val, {k: v.clone() for k, v in model.state_dict().items()}, it
            if it % (args.eval_every * 5) == 0:
                print(f'iter {it}: train loss {loss.item():.3f}, val error / baseline {val:.3f}, '
                      f'best {best:.3f} at {best_it} ({time.time() - started:.0f} s)')
            if it - best_it >= args.patience:
                stopped = f'no improvement in {args.patience} iterations'
                print(f'stopped at iteration {it}: {stopped}')
                break

    if best_state is None:
        sys.exit('no evaluation gave a finite val error (is --iters below --eval-every?): nothing saved')
    if not all(torch.isfinite(v).all() for v in best_state.values() if v.is_floating_point()):
        sys.exit('the best model has non-finite weights: nothing saved')
    model.load_state_dict(best_state)
    print(f'kept the model from iteration {best_it}')
    rows = evaluate(model, sessions, norm, args.device, part='test')
    print('\ntest (held out, not used to pick the model): model error / baseline error (under 1 is better than the baseline);')
    print('bump and slip: AUC, how well it ranks the steps they happen at (0.5 chance, 1.0 perfect)')
    print(f'{"":10s}' + ''.join(f'{h / 10:>8.1f} s' for h in HORIZONS))
    for name in list(TARGETS) + ['pose']:
        binary = name in TARGETS and TARGETS[name][1]
        cell = (lambda m, b: m) if binary else (lambda m, b: m / max(b, 1e-9))
        print(f'{name:10s}' + ''.join(f'{cell(*rows[(name, h)]):>10.2f}' for h in HORIZONS))
    results = {f'test_{h}': v for h, v in by_horizon(rows).items()}
    record.update(iterations=it, best_iteration=best_it, stopped=stopped, val_score=best, test_score=score(rows),
                  seconds=round(time.time() - started))
    if args.final_steps:
        final_paths = sorted(p for p in glob.glob(os.path.join(args.final_steps, '*.npz')) if not p.endswith('.maps.npz'))
        finals = [SessionData(dict(np.load(p))) for p in final_paths]
        if not final_paths or not sum(len(s.all) for s in finals):
            sys.exit(f'no usable sessions in {args.final_steps} for the final split')
        final_rows = evaluate(model, finals, norm, args.device, part='all')
        record['final_score'] = score(final_rows)
        record['final_sessions'] = runs.session_list(final_paths, finals)
        results.update({f'final_{h}': v for h, v in by_horizon(final_rows).items()})
        print(f'final ({len(final_paths)} sessions recorded after the search began): error / baseline {record["final_score"]:.3f}')
    record['results'] = results
    os.makedirs(args.out_dir, exist_ok=True)
    out = unused_path(args.out_dir, 'world')
    runs.save(out, {'model': {k: v.cpu() for k, v in model.state_dict().items()}, 'mean': norm.mean, 'std': norm.std,
                    'input_dim': sessions[0].inputs.shape[1], 'z': model.z_dim, 'genome': genome}, record)
    print(f'saved {out} (and its run record, .json)')


if __name__ == '__main__':
    main()
