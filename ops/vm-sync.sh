#!/usr/bin/env bash
# Bring the VM's ~/turtlebot up to date with GitHub, rebuild and run the unit tests, from the Mac:
#   ops/vm-sync.sh [branch]      (default: stay on the VM's current branch)
# Pulls fast-forward only. Refuses if the VM's checkout has local changes or a field test is
# running. Only pushed commits reach the VM.
set -euo pipefail

ssh -o BatchMode=yes turtlebot-vm bash -s -- "${1:-}" <<'REMOTE'
set -euo pipefail
branch=${1:-}  # SSH drops an empty argument
cd ~/turtlebot
if pgrep -f "ros2 launch pet_explorer" >/dev/null; then
  echo "error: a field test is running; stop it first (ops/field-test.sh stop)" >&2
  exit 1
fi
if [[ -n $(git status --porcelain) ]]; then
  echo "error: local changes in the VM's ~/turtlebot; not touching them:" >&2
  git status --short >&2
  exit 1
fi
git fetch -q origin
[[ -z $branch ]] || git checkout -q "$branch"
git pull -q --ff-only
echo "~/turtlebot on $(git branch --show-current) at $(git log --oneline -1)"
set +u
source /opt/ros/jazzy/setup.bash
set -u
colcon build --symlink-install 2>&1 | tail -2
colcon test --packages-select pet_explorer >/dev/null 2>&1 || true
colcon test-result --all
REMOTE
