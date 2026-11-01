"""The driving model (see MODEL.md): an encoder to a state z, latent dynamics, prediction heads, and
goal-conditioned values and policy on the same state."""
import torch
from torch import nn
import torch.nn.functional as F

from pet_learn.dataset import MASKS, TARGETS


def mlp(sizes, out_norm=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.ELU()]
    if out_norm:
        layers.append(nn.LayerNorm(sizes[-1]))
    return nn.Sequential(*layers)


class WorldModel(nn.Module):

    def __init__(self, input_dim: int, action_dim: int = 2, z_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.encoder = mlp([input_dim, hidden, hidden, z_dim], out_norm=True)
        self.dynamics = mlp([z_dim + action_dim, hidden, hidden, z_dim])
        self.dyn_norm = nn.LayerNorm(z_dim)
        self.heads = nn.ModuleDict({name: mlp([z_dim, hidden, size]) for name, (size, _) in TARGETS.items()})
        self.pose_head = mlp([z_dim * 2, hidden, 3])  # from the start state and the predicted one

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


def world_loss(model: WorldModel, batch: dict, bump_weight: float = 10.0, consistency: float = 1.0):
    """Prediction error of every head at every horizon (0..h), plus latent consistency. Returns the
    total and the parts."""
    x = batch['inputs']
    zs = model.rollout(model.encode(x[:, 0]), batch['actions'])  # (B, h+1, z)
    preds = model.decode(zs, {name: batch[name][:, :1] for name in TARGETS})
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
