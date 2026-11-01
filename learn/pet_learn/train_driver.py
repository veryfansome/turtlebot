"""Train the driving model's values and policy (IQL) jointly with its world model, and check them on
held-out driving.

    uv run python -m pet_learn.train_driver --world ../data/models/world-<time>.pt [--iters 20000]

Starts from a trained world model (pet_learn.train_world) and keeps training it alongside. Goals come
from pet_learn.goals: poses reached later (hindsight) and random poses in mapped free space. Checks,
on goals in the held-out last 20% of each session:

- values: how well V(state, goal) agrees with the recorded driving's actual discounted return
  (rank correlation; IQL aims above the recorded driving, so agreement in order is what counts);
- policy: what it does with the goal placed ahead, to the sides, back-left, back-right and straight
  behind real held-out states with open space ahead: it should drive forward, and turn toward the
  others. Straight behind, either way is as good, and the policy is symmetric, so in a symmetric
  situation it can't pick one: real situations are rarely symmetric.
"""
import argparse
import copy
import glob
import os
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from pet_learn.dataset import Normalizer, SessionData, mirror_inputs, window_batch
from pet_learn.episodes import usable
from pet_learn.goals import goal_features, relabel
from pet_learn.maps import SessionMaps
from pet_learn.model import WorldModel, mlp, world_loss
from pet_learn.train_world import to_torch

GOAL_DIM, ACTION_DIM = 5, 2
BEHIND = np.radians(120)  # a goal at a bearing beyond this is behind the robot
REVERSE_CAP = 0.15 / 0.3  # the safety layer's reverse limit, in the action's units (forward is scaled by 0.3 m/s)


GOAL_MIRROR = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])  # goal features flipped left to right (mirror_goal)
ACTION_MIRROR = torch.tensor([1.0, -1.0])  # forward stays, turn changes sign


class Driver(nn.Module):
    """Values and policy for a goal g (goals.goal_features), on a pair of world-model states s = (z, zm):
    the situation and its mirror image (encode_pair). Symmetric by construction: each output averages
    the networks' answer for the situation and, flipped back, for its mirror image. Driving is the same
    left and right, and this way the policy can't come to favour turning one way, whatever the data or
    the training noise."""

    def __init__(self, z_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = mlp([z_dim + GOAL_DIM + ACTION_DIM, hidden, hidden, 1])
        self.q2 = mlp([z_dim + GOAL_DIM + ACTION_DIM, hidden, hidden, 1])
        self.v = mlp([z_dim + GOAL_DIM, hidden, hidden, 1])
        self.actor = mlp([z_dim + GOAL_DIM, hidden, hidden, ACTION_DIM])
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -1.0))

    def q(self, s, g, a):
        x = torch.cat([s[0], g, a], dim=-1)
        xm = torch.cat([s[1], g * GOAL_MIRROR, a * ACTION_MIRROR], dim=-1)
        return ((self.q1(x) + self.q1(xm)) / 2).squeeze(-1), ((self.q2(x) + self.q2(xm)) / 2).squeeze(-1)

    def value(self, s, g):
        return ((self.v(torch.cat([s[0], g], dim=-1)) + self.v(torch.cat([s[1], g * GOAL_MIRROR], dim=-1))) / 2).squeeze(-1)

    def policy(self, s, g):
        """Mean action (scaled by the safety layer's caps, within -1..1) and its spread."""
        own = self.actor(torch.cat([s[0], g], dim=-1))
        mirrored = self.actor(torch.cat([s[1], g * GOAL_MIRROR], dim=-1)) * ACTION_MIRROR
        return torch.tanh((own + mirrored) / 2), self.log_std.exp()


def encode_pair(world, x, input_mirror):
    """The world model's states for inputs x and for their mirror image (x @ input_mirror)."""
    return world.encode(x), world.encode(x @ input_mirror)


def input_mirror(dim: int) -> torch.Tensor:
    """mirror_inputs as a matrix: it only reorders inputs and flips signs, so x @ this mirrors x."""
    return torch.as_tensor(mirror_inputs(np.eye(dim, dtype=np.float32)))


def mirror_goal(g):
    """Goal features flipped left to right: sideways offset and heading error change sign."""
    return g * np.array([1, -1, 1, -1, 1], np.float32)


def expectile(diff, tau):
    return (torch.abs(tau - (diff < 0).float()) * diff ** 2).mean()


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, '..', '..', 'data'))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', default=os.path.join(data_dir, 'steps'))
    parser.add_argument('--world', required=True, help='a world model checkpoint from pet_learn.train_world')
    parser.add_argument('--iters', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--goals', type=int, default=3000, help='training goals per session')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.7, help='IQL expectile')
    parser.add_argument('--beta', type=float, default=3.0, help='IQL advantage temperature')
    parser.add_argument('--reward-scale', type=float, default=10.0,
                        help="rewards are multiplied by this for training: one step's reward is a few hundredths, too "
                             'small for the advantage weights to tell actions apart')
    parser.add_argument('--behind-weight', type=float, default=3.0,
                        help='the policy counts steps with the goal behind the robot this many times: turning to '
                             'face a goal is rare in driving that mostly heads forward')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    paths = sorted(p for p in glob.glob(os.path.join(args.steps, '*.npz')) if not p.endswith('.maps.npz'))
    sessions, train_tr, val_tr = [], [], []
    for path in paths:
        steps = dict(np.load(path))
        data = SessionData(steps)
        maps = SessionMaps(path.replace('.npz', '.maps.npz'))
        ok = usable(steps)
        split = int(len(ok) * 0.8)
        sessions.append(data)
        train_tr.append(relabel(steps, maps, ok, 0, split, args.goals, rng, gamma=args.gamma))
        val_tr.append(relabel(steps, maps, ok, split, len(ok), max(args.goals // 4, 50), rng, gamma=args.gamma))
    print(f'{len(sessions)} sessions; transitions: {sum(len(t.step) for t in train_tr)} training, '
          f'{sum(len(t.step) for t in val_tr)} held out')

    ckpt = torch.load(args.world, weights_only=False)
    world = WorldModel(ckpt['input_dim'], z_dim=ckpt['z'], hidden=ckpt['hidden'])
    world.load_state_dict(ckpt['model'])
    norm = Normalizer(sessions)
    driver = Driver(ckpt['z'])
    target_q = copy.deepcopy(driver)
    opt = torch.optim.AdamW(list(world.parameters()) + list(driver.parameters()), lr=args.lr, weight_decay=1e-3)
    windows = [(i, s) for i, d in enumerate(sessions) for s in d.train]

    mirror_x = input_mirror(ckpt['input_dim'])

    def transition_batch(trs, idx_by_session):
        x0, x1, a, g0, g1, r, d, hs = [], [], [], [], [], [], [], []
        for i, idx in idx_by_session.items():
            t, data = trs[i], sessions[i]
            k = t.step[idx]
            x0.append(data.inputs[k]); x1.append(data.inputs[k + 1]); a.append(data.actions[k])
            g0.append(t.goal[idx]); g1.append(t.next_goal[idx]); r.append(t.reward[idx]); d.append(t.done[idx])
            hs.append(t.hindsight[idx].astype(np.float32))
        return [torch.as_tensor(np.concatenate(v)) for v in (x0, x1, a, g0, g1, r, d, hs)]

    def sample(trs, n):
        sizes = np.array([len(t.step) for t in trs])
        which = rng.choice(len(trs), n, p=sizes / sizes.sum())
        return {i: rng.integers(0, sizes[i], (which == i).sum()) for i in set(which.tolist())}

    started = time.time()
    for it in range(1, args.iters + 1):
        # World model, half mirrored.
        pick = rng.integers(0, len(windows), args.batch)
        by_session = {}
        for j in pick:
            by_session.setdefault(windows[j][0], []).append(windows[j][1])
        parts = [window_batch(sessions[i], np.array(st), norm, mirrored=bool(m))
                 for i, st in by_session.items() for m, st in enumerate(np.array_split(np.array(st), 2)) if len(st)]
        wloss, _ = world_loss(world, to_torch({k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, 'cpu'))

        # IQL (no need to mirror batches: the Driver is symmetric by construction).
        x0, x1, a, g0, g1, r, d, hs = transition_batch(train_tr, sample(train_tr, args.batch))
        s0 = encode_pair(world, x0, mirror_x)
        s0_fixed = tuple(z.detach() for z in s0)
        with torch.no_grad():
            s1 = encode_pair(world, x1, mirror_x)
            tq1, tq2 = target_q.q(s0, g0, a)
            target_v = driver.value(s1, g1)
        v = driver.value(s0_fixed, g0)
        v_loss = expectile(torch.min(tq1, tq2) - v, args.tau)
        q1, q2 = driver.q(s0, g0, a)
        q_target = args.reward_scale * r + args.gamma * (1 - d) * target_v
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        mean, std = driver.policy(s0_fixed, g0)
        with torch.no_grad():
            weight = torch.clamp(torch.exp(args.beta * (torch.min(tq1, tq2) - v)), max=100.0)
        # The policy learns the recorded commands within the safety layer's limits (the controller's turbo
        # goes past them), and only from hindsight goals; values and the world model see them as sent.
        target_a = torch.stack([a[:, 0].clamp(-REVERSE_CAP, 1.0), a[:, 1].clamp(-1.0, 1.0)], dim=-1)
        logp = torch.distributions.Normal(mean, std).log_prob(target_a).sum(-1)
        behind = torch.atan2(g0[:, 1], g0[:, 0]).abs() > BEHIND
        share = hs * torch.where(behind, args.behind_weight, 1.0)
        actor_loss = -(weight * logp * share).sum() / share.sum().clamp(min=1.0)

        loss = wloss + v_loss + q_loss + actor_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(world.parameters()) + list(driver.parameters()), 10.0)
        opt.step()
        with torch.no_grad():
            for p, tp in zip(driver.parameters(), target_q.parameters()):
                tp.mul_(0.995).add_(0.005 * p)
        if it % 2000 == 0:
            adv = (torch.min(tq1, tq2) - v).detach()
            print(f'iter {it}: world {wloss.item():.3f}, V {v_loss.item():.3f}, Q {q_loss.item():.3f}, '
                  f'actor {actor_loss.item():.3f}, advantage spread {adv.std().item():.3f}, '
                  f'weights {weight.min().item():.2f}..{weight.max().item():.2f} ({time.time() - started:.0f} s)')

    # Checks on held-out goals.
    world.eval(); driver.eval()
    with torch.no_grad():
        values, returns = [], []
        for i, t in enumerate(val_tr):
            s = encode_pair(world, torch.as_tensor(sessions[i].inputs[t.step]), mirror_x)
            values.append(driver.value(s, torch.as_tensor(t.goal)).numpy())
            returns.append(t.ret)
        values, returns = np.concatenate(values), np.concatenate(returns)
        print(f'\nheld-out values: rank correlation with the recorded return {spearman(values, returns):.2f} '
              f'({len(values)} transitions)')

        # Policy probes: real held-out states with open space ahead, from every session, goal 2 m away.
        x = []
        for data in sessions:
            open_ahead = np.flatnonzero((data.inputs[:, 27:33].min(axis=1) * 4.0 > 1.5))
            x.append(data.inputs[open_ahead[open_ahead >= int(len(data.inputs) * 0.8)]])
        x = torch.as_tensor(np.concatenate(x))
        # Each state and its mirror image, so the probes aren't swayed by which side happened to have
        # more room (the policy turns toward open space).
        x = torch.cat([x, x @ mirror_x])
        s = encode_pair(world, x, mirror_x)
        zero = np.zeros((len(x), 3))
        print(f'policy, goal 2 m away (forward m/s, turn rad/s; turn > 0 is left), median over {len(x)} held-out '
              'states with 1.5 m clear ahead and their mirror images:')
        for name, deg in [('ahead', 0), ('left', 90), ('right', -90), ('back-left', 150), ('back-right', -150),
                          ('behind', 180)]:
            goal = (2 * np.cos(np.radians(deg)), 2 * np.sin(np.radians(deg)), np.radians(deg))
            mean, _ = driver.policy(s, torch.as_tensor(goal_features(zero, goal)))
            fwd, turn = np.median(mean.numpy(), axis=0) * np.array([0.3, 1.5])
            print(f'  {name:10s} forward {fwd:+.2f}, turn {turn:+.2f}')

    os.makedirs(os.path.join(data_dir, 'models'), exist_ok=True)
    out = os.path.join(data_dir, 'models', f'driver-{time.strftime("%Y%m%d-%H%M%S")}.pt')
    torch.save({'world': world.state_dict(), 'driver': driver.state_dict(), 'mean': norm.mean, 'std': norm.std,
                'input_dim': ckpt['input_dim'], 'z': ckpt['z'], 'hidden': ckpt['hidden']}, out)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
