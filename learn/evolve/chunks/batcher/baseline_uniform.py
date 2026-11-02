"""Training batches: the baseline. Each step draws batch_size training windows uniformly at random
(with replacement) from all sessions, mirrors half of each session's share left to right
(dataset.mirror: physics is symmetric, so this doubles the data honestly), and hides each sensor in
a window with probability hide (dataset.hide_in_batch: the encoder sees it as missing, it stays a
target).

CONTRACT (batcher axis). Expose make(sessions, train, norm, rng, batch_size, **params) ->
next_batch(), where sessions are dataset.SessionData, train the (session index, window start)
pairs of every training window, norm the dataset.Normalizer, rng a numpy Generator seeded by the
run's seed. next_batch() returns one batch: a dict of numpy arrays as dataset.window_batch returns
them (standardized), with batch_size windows. Use only training windows (train), never val or test
ones; draw randomness only from rng, so a run is reproducible from its seed.
"""
import numpy as np

from pet_learn.dataset import window_batch


def make(sessions, train, norm, rng, batch_size, hide=0.05):
    def next_batch():
        pick = rng.integers(0, len(train), batch_size)
        by_session = {}
        for j in pick:
            by_session.setdefault(train[j][0], []).append(train[j][1])
        # Half of each batch mirrored left to right.
        parts = [window_batch(sessions[i], np.array(st), norm, mirrored=bool(m), rng=rng, hide_p=hide)
                 for i, st in by_session.items() for m, st in enumerate(np.array_split(np.array(st), 2)) if len(st)]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return next_batch
