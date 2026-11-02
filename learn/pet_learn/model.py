"""The driving model (see MODEL.md): an encoder to a state z, latent dynamics, prediction heads, and
goal-conditioned values and policy on the same state."""
import torch
from torch import nn
import torch.nn.functional as F

from pet_learn.dataset import MASKS, SENSORS, TARGETS


def mlp(sizes, out_norm=False, dropout=0.0):
    """Linear layers through sizes, each hidden one followed by LayerNorm, ELU and (only when dropout > 0,
    so models without it keep their layer numbering and random draws) dropout."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.ELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    if out_norm:
        layers.append(nn.LayerNorm(sizes[-1]))
    return nn.Sequential(*layers)


class WorldModel(nn.Module):

    def __init__(self, input_dim: int, action_dim: int = 2, z_dim: int = 128, hidden: int = 256,
                 enc_layers: int = 2, dyn_layers: int = 2, head_layers: int = 1, dropout: float = 0.0):
        """enc_layers, dyn_layers and head_layers: hidden layers (each `hidden` wide) in the encoder, the
        dynamics, and each prediction head (the pose head included); dropout after each hidden layer."""
        super().__init__()
        if min(enc_layers, dyn_layers, head_layers) < 1:
            raise ValueError('every part needs at least one hidden layer')
        if not 0 <= dropout < 1:
            raise ValueError(f'dropout must be in [0, 1), got {dropout}')
        self.encoder = mlp([input_dim] + [hidden] * enc_layers + [z_dim], out_norm=True, dropout=dropout)
        self.dynamics = mlp([z_dim + action_dim] + [hidden] * dyn_layers + [z_dim], dropout=dropout)
        self.dyn_norm = nn.LayerNorm(z_dim)
        self.heads = nn.ModuleDict({name: mlp([z_dim] + [hidden] * head_layers + [size], dropout=dropout)
                                    for name, (size, _) in TARGETS.items()})
        self.pose_head = mlp([z_dim * 2] + [hidden] * head_layers + [3], dropout=dropout)  # from the start state and the predicted one

    def encode(self, x):
        return self.encoder(x)

    def step(self, z, a):
        return self.dyn_norm(z + self.dynamics(torch.cat([z, a], dim=-1)))

    def rollout(self, z0, actions):
        """States z_0..z_h, chaining the dynamics on actions (B, h, 2)."""
        zs = [z0]
        for k in range(actions.shape[1]):
            zs.append(self.step(zs[-1], actions[:, k]))
        return torch.stack(zs, dim=1)

    def decode(self, z, current: dict):
        """Predictions from states z (B, h+1, z). Continuous readings are predicted as a change from
        their current values (current[name]: (B, 1, size)), so "nothing changes" is the default the
        model improves on; bump and slip are logits."""
        out = {}
        for name, head in self.heads.items():
            out[name] = head(z) if TARGETS[name][1] else current[name] + head(z)
        return out

    def pose(self, z0, zk):
        return self.pose_head(torch.cat([z0.expand_as(zk), zk], dim=-1))


def current_readings(batch: dict) -> dict:
    """Each target's reading at the windows' first step (B, 1, size), as the encoder saw it: a reading it
    couldn't see (missing or hidden) is 0, the mean in standardized units, so predictions can't lean on it."""
    return {name: batch[name][:, :1] * batch[f'{name}_seen'][:, :1] if name in SENSORS else batch[name][:, :1]
            for name in TARGETS}


def world_loss(model: WorldModel, batch: dict, bump_weight: float = 10.0, consistency: float = 1.0):
    """Prediction error of every head at every horizon (0..h), plus latent consistency. Returns the
    total and the parts."""
    x = batch['inputs']
    zs = model.rollout(model.encode(x[:, 0]), batch['actions'])  # (B, h+1, z)
    preds = model.decode(zs, current_readings(batch))
    parts = {}
    for name, (size, binary) in TARGETS.items():
        target = batch[name]
        if binary:
            weight = torch.where(target > 0.5, bump_weight, 1.0) if name == 'bump' else None
            parts[name] = F.binary_cross_entropy_with_logits(preds[name], target, weight=weight)
        elif name in MASKS:  # only where the reading was recorded
            mask = batch[MASKS[name]]
            parts[name] = (((preds[name] - target) ** 2) * mask).sum() / (mask.sum() * size).clamp(min=1.0)
        else:
            parts[name] = F.mse_loss(preds[name], target)
    pose = model.pose(zs[:, :1], zs[:, 1:])
    parts['pose'] = F.mse_loss(pose, batch['pose'][:, 1:])
    with torch.no_grad():
        target_z = model.encode(x[:, 1:])
    parts['consistency'] = consistency * F.mse_loss(zs[:, 1:], target_z)
    return sum(parts.values()), parts
