"""Optimizer: the baseline, AdamW at a constant learning rate 6e-4, weight decay 1e-3, batches of
1024 windows (on five sessions this beat batch 256 at 3e-4 on val error, 2026-11-01).

CONTRACT (optim axis). Expose make(parameters, **params) -> (optimizer, scheduler or None,
batch_size). The harness calls optimizer.step() after clipping gradients to norm 10, then
scheduler.step() if there is one, every iteration. Training stops when the val error hasn't
improved for 2,000 iterations (or at 50,000), so a schedule can't know the number of steps in
advance: make it a function of the step count, not of a planned total. batch_size is the number
of training windows per step.
"""
import torch


def make(parameters, lr=6e-4, weight_decay=1e-3, batch=1024):
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay), None, batch
