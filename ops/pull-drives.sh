#!/usr/bin/env bash
# Copy recorded driving sessions from the VM to the Mac, for training:
#   ops/pull-drives.sh
# Copies new bags from ~/drives/ on the VM into data/drives/ in the repo (git-ignored). Only copies
# what's missing or changed, and never deletes anything on either side. Skips a session that's still
# being recorded.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
dest=$repo/data/drives
mkdir -p "$dest"

ssh -o BatchMode=yes turtlebot-vm 'test -d ~/drives' || { echo "no sessions recorded on the VM yet"; exit 0; }
recording=$(ssh -o BatchMode=yes turtlebot-vm \
  'f=~/field-tests/run.pid; [[ -f $f ]] && kill -0 "$(cat $f)" 2>/dev/null && basename "$(ls -t ~/field-tests/*.log | head -1)" .log || true')
exclude=()
[[ -z $recording ]] || { echo "skipping $recording: still recording"; exclude=(--exclude "/$recording/"); }

rsync -a ${exclude[@]+"${exclude[@]}"} turtlebot-vm:drives/ "$dest/"
echo "sessions in $dest:"
du -sh "$dest"/*/ 2>/dev/null | sed 's|'"$dest"'/||' || echo "  none yet"
