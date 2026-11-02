"""The data the world-model search is measured on, pinned: every steps file's name and fingerprint.

    uv run python -m eval.manifest --write    # pin what's in $PET_STEPS (default ../data/steps) now

Scores mean something only against fixed data, so the search's eval refuses to run on anything else
(eval.preflight). New recordings don't join it: they are the final split (PET_FINAL_STEPS). Re-pinning
starts a new measurement epoch: every archived score was against the old data and must be rescored.
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, 'data_manifest.json')


def steps_files(folder: str) -> list:
    return sorted(p for p in glob.glob(os.path.join(folder, '*.npz')) if not p.endswith('.maps.npz'))


def describe(folder: str) -> dict:
    from pet_learn.runs import fingerprint
    return {os.path.basename(p): fingerprint(p) for p in steps_files(folder)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    folder = os.environ.get('PET_STEPS', os.path.join(HERE, '..', '..', 'data', 'steps'))
    pinned = describe(folder)
    print(json.dumps(pinned, indent=1))
    if args.write:
        with open(MANIFEST, 'w') as f:
            json.dump({'sessions': pinned}, f, indent=1)
            f.write('\n')
        print(f'wrote {MANIFEST}')


if __name__ == '__main__':
    main()
