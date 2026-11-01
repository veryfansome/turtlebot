"""Train the driving model's values and policy (IQL) jointly with its world model, and check them on
held-out driving.

    uv run python -m pet_learn.train_driver --world ../data/models/world-<time>.pt [--iters 20000]

Starts from a trained world model (pet_learn.train_world) and keeps training it alongside. Goals come
from pet_learn.goals: poses reached later (hindsight) and random poses in mapped free space. Checks,
on goals in the held-out last 20% of each session:

- values: how well V(state, goal) agrees with the recorded driving's actual discounted return
  (rank correlation; IQL aims above the recorded driving, so agreement in order is what counts);
- policy: what it does with the goal placed straight ahead, behind, left and right of real held-out
  states with open space ahead: it should drive forward, turn around, turn left and turn right.
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


class Driver(nn.Module):
    """Values and policy on the world model's state z, for a goal g (goals.goal_features)."""

    def __init__(self, z_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = mlp([z_dim + GOAL_DIM + ACTION_DIM, hidden, hidden, 1])
        self.q2 = mlp([z_dim + GOAL_DIM + ACTION_DIM, hidden, hidden, 1])
        self.v = mlp([z_dim + GOAL_DIM, hidden, hidden, 1])
        self.actor = mlp([z_dim + GOAL_DIM, hidden, hidden, ACTION_DIM])
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -1.0))

    def q(self, z, g, a):
        x = torch.cat([z, g, a], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def value(self, z, g):
        return self.v(torch.cat([z, g], dim=-1)).squeeze(-1)

    def policy(self, z, g):
        """Mean action (scaled by the safety layer's caps, within -1..1) and its spread."""
        return torch.tanh(self.actor(torch.cat([z, g], dim=-1))), self.log_std.exp()


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

    def transition_batch(trs, idx_by_session, mirrored):
        x0, x1, a, g0, g1, r, d = [], [], [], [], [], [], []
        for i, idx in idx_by_session.items():
            t, data = trs[i], sessions[i]
            k = t.step[idx]
            x0.append(data.inputs[k]); x1.append(data.inputs[k + 1]); a.append(data.actions[k])
            g0.append(t.goal[idx]); g1.append(t.next_goal[idx]); r.append(t.reward[idx]); d.append(t.done[idx])
        b = [np.concatenate(v) for v in (x0, x1, a, g0, g1, r, d)]
        if mirrored:
            b[0], b[1] = mirror_inputs(b[0]), mirror_inputs(b[1])
            b[2] = b[2] * np.array([1, -1], np.float32)
            b[3], b[4] = mirror_goal(b[3]), mirror_goal(b[4])
        return [torch.as_tensor(v) for v in b]

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

        # IQL.
        x0, x1, a, g0, g1, r, d = transition_batch(train_tr, sample(train_tr, args.batch), mirrored=it % 2 == 0)
        z0 = world.encode(x0)
        with torch.no_grad():
            z1 = world.encode(x1)
            tq1, tq2 = target_q.q(z0, g0, a)
            target_v = driver.value(z1, g1)
        v = driver.value(z0.detach(), g0)
        v_loss = expectile(torch.min(tq1, tq2) - v, args.tau)
        q1, q2 = driver.q(z0, g0, a)
        q_target = args.reward_scale * r + args.gamma * (1 - d) * target_v
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        mean, std = driver.policy(z0.detach(), g0)
        with torch.no_grad():
            weight = torch.clamp(torch.exp(args.beta * (torch.min(tq1, tq2) - v)), max=100.0)
        logp = torch.distributions.Normal(mean, std).log_prob(a).sum(-1)
        actor_loss = -(weight * logp).mean()

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
            z = world.encode(torch.as_tensor(sessions[i].inputs[t.step]))
            values.append(driver.value(z, torch.as_tensor(t.goal)).numpy())
            returns.append(t.ret)
        values, returns = np.concatenate(values), np.concatenate(returns)
        print(f'\nheld-out values: rank correlation with the recorded return {spearman(values, returns):.2f} '
              f'({len(values)} transitions)')

        # Policy probes: real held-out states with open space ahead, goal 2 m away in four directions.
        print('policy, goal 2 m away (forward m/s, turn rad/s; turn > 0 is left), median over held-out states with 1.5 m clear ahead:')
        for i, data in enumerate(sessions):
            open_ahead = np.flatnonzero((data.inputs[:, 27:33].min(axis=1) * 4.0 > 1.5))
            open_ahead = open_ahead[open_ahead >= int(len(data.inputs) * 0.8)]
            if len(open_ahead) == 0:
                continue
            z = world.encode(torch.as_tensor(data.inputs[open_ahead]))
            zero = np.zeros((len(open_ahead), 3))
            for name, goal in [('ahead', (2, 0, 0)), ('behind', (-2, 0, np.pi)), ('left', (0, 2, np.pi / 2)),
                               ('right', (0, -2, -np.pi / 2))]:
                g = torch.as_tensor(goal_features(zero, goal))
                mean, _ = driver.policy(z, g)
                fwd, turn = np.median(mean.numpy(), axis=0) * np.array([0.3, 1.5])
                print(f'  {name:6s} forward {fwd:+.2f}, turn {turn:+.2f}   ({len(open_ahead)} states)')
            break

    os.makedirs(os.path.join(data_dir, 'models'), exist_ok=True)
    out = os.path.join(data_dir, 'models', f'driver-{time.strftime("%Y%m%d-%H%M%S")}.pt')
    torch.save({'world': world.state_dict(), 'driver': driver.state_dict(), 'mean': norm.mean, 'std': norm.std,
                'input_dim': ckpt['input_dim'], 'z': ckpt['z'], 'hidden': ckpt['hidden']}, out)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
