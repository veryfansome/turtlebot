#!/usr/bin/env bash
# Run an ops unit on the Pi or the VM, from the Mac:
#   ops/run.sh <pi|vm> <unit> [status|diff|apply|revert]
# status and diff are read-only and run as the login user. apply and revert run with sudo: in a
# terminal, sudo asks for the password; without one (e.g. from Claude Code) they run only where
# sudo needs no password, and otherwise stop and print the command to run yourself.
set -euo pipefail

die() { echo "error: $*" >&2; exit 2; }

ops_dir=$(cd "$(dirname "$0")" && pwd)
[[ $# -ge 2 ]] || die "usage: ops/run.sh <pi|vm> <unit> [status|diff|apply|revert]"
target=$1 unit=$2 action=${3:-status}
case $target in
  pi) host=turtlebot-pi ;;
  vm) host=turtlebot-vm ;;
  *) die "target must be pi or vm" ;;
esac
[[ -f $ops_dir/$target/$unit.sh ]] \
  || die "no unit '$unit' for $target (have: $(cd "$ops_dir/$target" && ls -- *.sh 2>/dev/null | sed 's/\.sh$//' | tr '\n' ' '))"
case $action in status|diff|apply|revert) ;; *) die "action must be status, diff, apply or revert" ;; esac

remote=/tmp/turtlebot-ops
ssh -o BatchMode=yes "$host" "mkdir -p $remote/$target"
scp -q "$ops_dir/lib.sh" "$host:$remote/lib.sh"
scp -q "$ops_dir/$target/$unit.sh" "$host:$remote/$target/$unit.sh"
cmd="bash $remote/$target/$unit.sh $action"

case $action in
  status|diff) exec ssh -o BatchMode=yes "$host" "$cmd" ;;
esac
if [[ -t 0 ]]; then
  exec ssh -t "$host" "sudo $cmd"
elif ssh -o BatchMode=yes "$host" sudo -n true 2>/dev/null; then
  exec ssh -o BatchMode=yes "$host" "sudo -n $cmd"
else
  echo "$action needs the sudo password on $host. Run it in a terminal:" >&2
  echo "  ops/run.sh $target $unit $action" >&2
  exit 3
fi
