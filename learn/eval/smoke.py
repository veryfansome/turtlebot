"""eval.smoke for the world-model search: a cheap correctness gate before any scored run.

    python -m eval.smoke GENOME

On two small real sessions: the genome's impls load and honour their axes' contracts (the baseline
files' docstrings): the model has z_dim and the encode/rollout/decode/pose interface with the right
shapes, the objective gives a finite scalar with gradients, the optimizer trains, the batcher gives
batch_size windows with the right keys and is reproducible from its seed, and the harness can
evaluate the result. Exit 1 with the reason if not.
"""
import sys

import numpy as np
import torch

from eval.common import small_sessions
from pet_learn import genome as genomes
from pet_learn.dataset import TARGETS
from pet_learn.train_world import default_device, evaluate, score, to_torch


def check(ok: bool, why: str):
    if not ok:
        print(f'smoke: {why}', file=sys.stderr)
        sys.exit(1)


def main():
    genome = genomes.load(sys.argv[1])
    device = default_device()
    torch.manual_seed(0)
    sessions, train, norm = small_sessions()
    arch, ap = genomes.impl(genome, 'arch')
    objective, objp = genomes.impl(genome, 'objective')
    optim, opp = genomes.impl(genome, 'optim')
    batcher, bp = genomes.impl(genome, 'batcher')
    model = arch.build(sessions[0].inputs.shape[1], **ap).to(device)
    check(isinstance(getattr(model, 'z_dim', None), int), 'the model needs an int z_dim')
    for name in ('encode', 'rollout', 'decode', 'pose'):
        check(callable(getattr(model, name, None)), f'the model needs {name}()')
    opt, scheduler, batch_size = optim.make(model.parameters(), **opp)
    check(isinstance(batch_size, int) and batch_size > 0, f'optim must return a positive int batch size, not {batch_size!r}')
    batch_size = min(batch_size, 256)  # the smoke needs only a few windows
    one = batcher.make(sessions, train, norm, np.random.default_rng(7), batch_size, **bp)()
    two = batcher.make(sessions, train, norm, np.random.default_rng(7), batch_size, **bp)()
    check(set(one) == set(two) and all(np.array_equal(one[k], two[k], equal_nan=True) for k in one),
          'the batcher must be reproducible from its rng')
    check(len(one['inputs']) == batch_size, f'the batcher gave {len(one["inputs"])} windows, not {batch_size}')
    batch = to_torch(one, device)
    zs = model.rollout(model.encode(batch['inputs'][:, 0]), batch['actions'])
    check(tuple(zs.shape) == (batch_size, batch['inputs'].shape[1], model.z_dim),
          f'rollout gave {tuple(zs.shape)}, expected (B, h+1, z_dim)')
    preds = model.decode(zs, {n: batch[n][:, :1] for n in TARGETS})
    check(set(preds) == set(TARGETS), f'decode must predict every target: {sorted(TARGETS)}')
    for n in TARGETS:
        check(tuple(preds[n].shape) == tuple(batch[n].shape), f'decode {n}: {tuple(preds[n].shape)}, expected {tuple(batch[n].shape)}')
    pose = model.pose(zs[:, :1], zs[:, 1:])
    check(tuple(pose.shape) == tuple(batch['pose'][:, 1:].shape), f'pose: {tuple(pose.shape)}, expected (B, h, 3)')
    for _ in range(3):
        loss, parts = objective.loss(model, batch, **objp)
        check(torch.is_tensor(loss) and loss.dim() == 0 and loss.requires_grad, 'the objective must return a scalar carrying grad')
        check(bool(torch.isfinite(loss)), f'the objective is not finite: {loss.item()}')
        check(isinstance(parts, dict), 'the objective must return (total, parts dict)')
        opt.zero_grad()
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
    val = score(evaluate(model, sessions, norm, device))
    check(np.isfinite(val), f'the trained model scores {val} on val windows')
    print(f'smoke ok: {genomes.recipe(genome)}; loss {loss.item():.3f}, val error / baseline {val:.3f} after 3 steps')


if __name__ == '__main__':
    main()
