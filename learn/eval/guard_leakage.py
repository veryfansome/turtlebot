"""eval.guardrails for the world-model search: the batcher trains only on training windows.

    python -m eval.guard_leakage GENOME

The batcher is handed the sessions whole (it needs them to cut windows), so nothing stops it reading
the val or test part, which pick and report the model. This hands it copies in which every step no
training window covers is NaN, then draws batches: any NaN in what it returns came from held-out
driving. (Sensors that are really missing are 0 with their flag off, never NaN.) Exit 1 if so.
"""
import copy
import sys

import numpy as np

from eval.common import small_sessions
from pet_learn import genome as genomes


def poisoned(data):
    """A copy of a SessionData with every step outside the training windows set to NaN."""
    covered = np.zeros(len(data.inputs), bool)
    for s in data.train:
        covered[s:s + data.h + 1] = True
    out = copy.copy(data)
    out.inputs = data.inputs.copy()
    out.inputs[~covered] = np.nan
    out.actions = data.actions.copy()
    out.actions[~covered] = np.nan
    out.targets = {k: v.astype(np.float32).copy() for k, v in data.targets.items()}
    for v in out.targets.values():
        v[~covered] = np.nan
    out.pose = data.pose.copy()
    out.pose[~covered] = np.nan
    return out


def main():
    genome = genomes.load(sys.argv[1])
    sessions, train, norm = small_sessions()
    batcher, params = genomes.impl(genome, 'batcher')
    optim, opp = genomes.impl(genome, 'optim')
    import torch
    _, _, batch_size = optim.make([torch.zeros(1, requires_grad=True)], **opp)
    next_batch = batcher.make([poisoned(s) for s in sessions], train, norm, np.random.default_rng(0), batch_size, **params)
    for k in range(20):
        batch = next_batch()
        bad = sorted(key for key, v in batch.items() if np.issubdtype(v.dtype, np.floating) and np.isnan(v).any())
        if bad:
            print(f'guard_leakage: batch {k} carries held-out (val/test) driving in {bad}: the batcher must use '
                  'only the training windows it is given', file=sys.stderr)
            sys.exit(1)
    print('guard_leakage ok: 20 batches, training windows only')


if __name__ == '__main__':
    main()
