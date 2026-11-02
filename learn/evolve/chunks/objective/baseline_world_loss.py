"""Training objective: the baseline, pet_learn.model.world_loss: squared error of every continuous
head at every horizon (masked where the reading is missing), classification loss for bump (bumps
weighted 10x) and slip, squared error of the pose change, plus a latent consistency term.

CONTRACT (objective axis). Expose loss(model, batch) -> (total, parts): total a scalar tensor
carrying grad, parts a dict of named scalar tensors (for logging). batch is one standardized
training batch from the batcher (dataset.raw_batch's keys, standardized by dataset.Normalizer:
inputs (B, h+1, d), actions (B, h, 2), every dataset.TARGETS name (B, h+1, size), pose
(B, h+1, 3), and for each sensor <name>_ok (the reading is real) and <name>_seen (the encoder saw
it) (B, h+1, 1)). Only real readings (<name>_ok) may count as targets; a reading the encoder
didn't see must not seed a prediction (model.current_readings). Selection measures held-out
prediction error relative to naive baselines, never this loss, so a loss that is low without
predicting well earns nothing.
"""
from pet_learn.model import world_loss


def loss(model, batch):
    return world_loss(model, batch)
