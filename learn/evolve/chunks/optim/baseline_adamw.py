"""Optimizer: the baseline, AdamW at a constant learning rate 6e-4 (optionally with warmup and an
exponential decay; both off by default), weight decay 1e-3, batches of 1024 windows (on five
sessions this beat batch 256 at 3e-4 on val error, 2026-11-01).

CONTRACT (optim axis). Expose make(parameters, **params) -> (optimizer, scheduler or None,
batch_size). The harness calls optimizer.step() after clipping gradients to norm 10, then
scheduler.step() if there is one, every iteration. Training stops when the val error hasn't
improved for 2,000 iterations (or at 50,000), so a schedule can't know the number of steps in
advance: make it a function of the step count, not of a planned total. batch_size is the number
of training windows per step.
"""
import torch


def make(parameters, lr=6e-4, weight_decay=1e-3, batch=1024, warmup=0, decay_halflife=0):
    """warmup: iterations of linear warmup to lr; decay_halflife: after warmup, the learning rate halves
    every this many iterations. Both 0 (the default): a constant lr and no scheduler."""
    opt = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if not warmup and not decay_halflife:
        return opt, None, batch
    if warmup < 0 or decay_halflife < 0:
        raise ValueError('warmup and decay_halflife must be >= 0')

    def factor(step):  # step counts completed iterations (0 before the first)
        if step < warmup:
            return min(1.0, (step + 1) / warmup)
        return 0.5 ** ((step - warmup) / decay_halflife) if decay_halflife else 1.0
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, factor), batch
