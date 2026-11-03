"""Training objective: the baseline, pet_learn.model.world_loss: squared error of every continuous
head at every horizon (masked where the reading is missing), classification loss for bump (bumps
weighted 10x, bump_pos_weight) and slip, squared error of the pose change, plus a latent
consistency term; each head's term can be weighted (<head>_weight, 1 by default).

CONTRACT (objective axis). Expose loss(model, batch, **params) -> (total, parts): total a scalar tensor
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


def loss(model, batch, consistency=1.0, bump_pos_weight=10.0,
         scan_weight=1.0, floor_weight=1.0, odom_weight=1.0, wheels_weight=1.0, current_weight=1.0,
         bump_weight=1.0, slip_weight=1.0, ir_weight=1.0, pose_weight=1.0):
    """consistency: the weight of the latent consistency term (predicted states against the encoder's).
    bump_pos_weight: how much the steps a bump happens at count within the bump head's loss.
    <head>_weight: scales that head's whole term (pose: the pose change). At their defaults this is the
    loss from before these params existed, bit for bit."""
    weights = {'scan': scan_weight, 'floor': floor_weight, 'odom': odom_weight, 'wheels': wheels_weight,
               'current': current_weight, 'bump': bump_weight, 'slip': slip_weight, 'ir': ir_weight,
               'pose': pose_weight}
    return world_loss(model, batch, bump_weight=bump_pos_weight, consistency=consistency, head_weights=weights)
