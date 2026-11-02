"""eval.setup for the world-model search: is this a working environment for scoring? Runs once per
scoring clone. A failure here is the environment's, not a candidate's (the engine doesn't archive it).

Checks: PET_STEPS holds exactly the pinned data (data_manifest.json); PET_FINAL_STEPS, if set, holds
only sessions outside it (the final split must be driving the search never trained on); torch imports
and the device trains.
"""
import json
import os
import sys

from eval.manifest import MANIFEST, describe


def fail(why: str):
    print(f'preflight: {why}', file=sys.stderr)
    sys.exit(1)


def main():
    steps = os.environ.get('PET_STEPS')
    if not steps or not os.path.isdir(steps):
        fail('PET_STEPS must name the folder of converted sessions (an absolute path: scoring runs in a copy)')
    with open(MANIFEST) as f:
        pinned = json.load(f)['sessions']
    have = describe(steps)
    if have != pinned:
        missing = sorted(set(pinned) - set(have))
        extra = sorted(set(have) - set(pinned))
        changed = sorted(n for n in set(pinned) & set(have) if pinned[n] != have[n])
        fail(f'{steps} is not the pinned data: missing {missing}, extra {extra}, changed {changed} '
             '(eval/data_manifest.json; re-pin with eval.manifest --write, which starts a new epoch)')
    final = os.environ.get('PET_FINAL_STEPS')
    if final:
        overlap = sorted(set(describe(final)) & set(pinned))
        if overlap:
            fail(f'PET_FINAL_STEPS shares sessions with the searched data: {overlap}')
    import torch
    from pet_learn.train_world import default_device
    device = default_device()
    x = torch.ones(4, 4, device=device, requires_grad=True)
    (x * 2).sum().backward()
    print(f'preflight ok: {len(have)} pinned sessions, device {device}, torch {torch.__version__}')


if __name__ == '__main__':
    main()
