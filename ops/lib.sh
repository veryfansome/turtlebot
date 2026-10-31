# Shared helpers for ops units. Sourced by each unit, which runs on the target machine (Pi or VM).
#
# A unit changes one file through a pair of transforms: unit_apply_transform and
# unit_revert_transform (stdin -> stdout, each a no-op on its own output). So `status` can tell
# applied from not applied by content alone, and `apply` and `revert` are safe to repeat. After a
# change the unit restarts what it needs and verifies the machine; if that fails, the previous
# file is restored and the restart repeated.
#
# Every change is logged to $OPS_LOG, and the file is backed up first: <name>.orig the first time
# any unit touches it (never overwritten), plus a timestamped copy each time.
set -euo pipefail

OPS_STATE_DIR=/var/lib/turtlebot-ops
OPS_LOG=$OPS_STATE_DIR/changes.log

ops_log() {  # unit action detail
  mkdir -p "$OPS_STATE_DIR"
  printf '%s %s %s %s\n' "$(date -Is)" "$1" "$2" "$3" >> "$OPS_LOG"
}

ops_backup() {  # unit file
  local dir=$OPS_STATE_DIR/backups/$1 name
  name=$(echo "$2" | tr / _)
  mkdir -p "$dir"
  [[ -f $dir/$name.orig ]] || cp -p "$2" "$dir/$name.orig"
  cp -p "$2" "$dir/$name.$(date +%Y%m%dT%H%M%S)"
}

ops_write() {  # file new_contents_file: replace atomically, keeping owner and mode
  local tmp
  tmp=$(mktemp "$1.ops.XXXXXX")
  cat "$2" > "$tmp"
  chmod --reference="$1" "$tmp"
  chown --reference="$1" "$tmp"
  mv "$tmp" "$1"
}

ops_wait_for() {  # description timeout_seconds command...
  local desc=$1 deadline=$((SECONDS + $2))
  shift 2
  until "$@" >/dev/null 2>&1; do
    if ((SECONDS >= deadline)); then
      echo "  ✗ $desc"
      return 1
    fi
    sleep 3
  done
  echo "  ✓ $desc"
}

ros_env() {  # the robot's own ROS environment (its setup files reference unset variables)
  set +u
  # shellcheck disable=SC1091
  source "${ROBOT_SETUP:-/etc/turtlebot4/setup.bash}"
  set -u
}

ops_main() {  # action
  local action=${1:-status}
  work=$(mktemp -d)
  trap 'rm -rf "$work"' EXIT
  cp "$TARGET_FILE" "$work/current"
  unit_apply_transform < "$work/current" > "$work/applied"
  unit_revert_transform < "$work/current" > "$work/reverted"
  case $action in
    status)
      if cmp -s "$work/current" "$work/applied"; then echo "$UNIT: applied"
      elif cmp -s "$work/current" "$work/reverted"; then echo "$UNIT: not applied"
      else echo "$UNIT: partly applied"
      fi ;;
    diff)
      diff -u --label "$TARGET_FILE (now)" --label "$TARGET_FILE (after apply)" \
        "$work/current" "$work/applied" || true ;;
    apply) _ops_change apply "$work/applied" unit_verify_apply ;;
    revert) _ops_change revert "$work/reverted" unit_verify_revert ;;
    *) echo "usage: $0 status|diff|apply|revert" >&2; exit 2 ;;
  esac
}

_ops_change() {  # action new_file verify_function
  local action=$1 new=$2 verify=$3
  if cmp -s "$work/current" "$new"; then
    echo "$UNIT: nothing to $action"
    return
  fi
  [[ $EUID -eq 0 ]] || { echo "error: $action needs root" >&2; exit 1; }
  ops_backup "$UNIT" "$TARGET_FILE"
  ops_write "$TARGET_FILE" "$new"
  ops_log "$UNIT" "$action" "$TARGET_FILE"
  echo "$UNIT: $action: wrote $TARGET_FILE; verifying"
  if unit_after_change && "$verify"; then
    echo "$UNIT: $action done"
    return
  fi
  echo "$UNIT: verification failed; restoring the previous $TARGET_FILE" >&2
  ops_write "$TARGET_FILE" "$work/current"
  ops_log "$UNIT" "$action-rolled-back" "$TARGET_FILE"
  unit_after_change || true
  exit 1
}
