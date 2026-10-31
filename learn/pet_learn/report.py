"""Report how well each recorded session drove, as goal-reaching episodes.

    uv run python -m pet_learn.report [--steps ../data/steps] [--episodes 300] [--seed 0]

A report, not a ranking: some numbers are better higher (success, SPL), some lower (time, energy,
bumps, stuck), and judging drivers is done by reading them. Recorded sessions without goals are cut
into hindsight episodes (pet_learn.episodes), so for a person driving, success is near 100% by
construction and SPL measures how direct their driving was.
"""
import argparse
import glob
import os

import numpy as np

from pet_learn.episodes import hindsight_episodes
from pet_learn.maps import SessionMaps


def session_report(name: str, steps: dict, episodes) -> str:
    minutes = steps['t'][-1] / 60
    driver = 'controller' if steps['joy_active'].mean() > 0.5 else 'driver'
    if not episodes:
        return f'{name} ({driver}, {minutes:.1f} min): no usable episodes'
    success = np.array([e.success for e in episodes])
    spl = np.array([e.spl for e in episodes])
    shortest = np.array([e.shortest for e in episodes])
    per_m = lambda values: np.sum(values) / max(shortest.sum(), 1e-9)
    rescues = int((np.diff(steps['kidnapped'].astype(int)) == 1).sum())
    stuck = ((steps['cmd'][:, 0] > 0.03) & (steps['mouse_speed'] < 0.01) & ~steps['docked']).mean()
    return '\n'.join([
        f'{name} ({driver}, {minutes:.1f} min, {len(episodes)} episodes, shortest paths '
        f'{np.median(shortest):.1f} m median)',
        f'  success {success.mean() * 100:.0f}%, SPL mean {spl.mean():.2f} (median {np.median(spl):.2f})',
        f'  per metre of shortest path: {per_m([e.duration for e in episodes]):.1f} s, '
        f'{per_m([e.energy for e in episodes]):.2f} A·s',
        f'  reward per episode {np.mean([e.reward for e in episodes]):+.2f} '
        f'(per metre {per_m([e.reward for e in episodes]):+.2f})',
        f'  session: bumps {steps["bump"].any(axis=1).sum() / minutes:.1f}/min, '
        f'stuck {stuck * 100:.1f}% of the time, rescues {rescues}',
    ])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_steps = os.path.normpath(os.path.join(here, '..', '..', 'data', 'steps'))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', default=default_steps)
    parser.add_argument('--episodes', type=int, default=300, help='hindsight episodes per session')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(args.steps, '*.npz')) if not p.endswith('.maps.npz'))
    if not paths:
        print(f'no converted sessions in {args.steps}; run pet_learn.convert')
    for path in paths:
        name = os.path.basename(path)[:-len('.npz')]
        steps = dict(np.load(path))
        maps = SessionMaps(path.replace('.npz', '.maps.npz'))
        episodes = hindsight_episodes(steps, maps, args.episodes, np.random.default_rng(args.seed))
        print(session_report(name, steps, episodes))


if __name__ == '__main__':
    main()
