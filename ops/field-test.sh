#!/usr/bin/env bash
# Run the explorer stack (SLAM, Nav2, explorer) on the VM in the background, from the Mac:
#   ops/field-test.sh start [launch args...]  preflight checks, then start (DRIVES THE ROBOT)
#   ops/field-test.sh status                  running or not, and the latest explorer lines
#   ops/field-test.sh logs [lines]            the end of the latest run's log (default 40 lines)
#   ops/field-test.sh save-map                save the SLAM map to ~/maps/<run>/ on the VM
#   ops/field-test.sh stop                    save the map, Ctrl-C the stack (the explorer cancels its goal), make sure it's gone
#   ops/field-test.sh estop [on|off]          the base's e-stop: on cuts drive motor power (default), off restores it
# Logs are kept in ~/field-tests/ on the VM, maps in ~/maps/. stop and estop are always safe to run.
set -euo pipefail

ssh -o BatchMode=yes turtlebot-vm bash -s -- "$@" <<'REMOTE'
set -euo pipefail
action=${1:-status}
shift || true
set +u
source /opt/ros/jazzy/setup.bash
source ~/turtlebot/install/setup.bash
set -u
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp  # as ~/.bashrc sets them (vm/provision.sh)

dir=~/field-tests
pidfile=$dir/run.pid
mkdir -p "$dir"
running() { [[ -f $pidfile ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; }
latest_log() { ls -t "$dir"/*.log 2>/dev/null | head -1 || true; }
first() { timeout 8 ros2 topic echo --once "$1" --field "$2" 2>/dev/null | head -1; }

# Save the running SLAM map next to the run's name: map.pgm/map.yaml (for Nav2 to navigate on) and
# map.posegraph/map.data (slam_toolbox's full state, to continue mapping from it later).
save_map() {
  local log name out
  log=$(latest_log)
  name=$(basename "${log:-manual}" .log)
  out=~/maps/$name
  mkdir -p "$out"
  timeout 30 ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: '$out/map'}}" \
    | grep -q "result=0" && echo "saved $out/map.pgm, map.yaml" || echo "warning: save_map failed" >&2
  timeout 30 ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '$out/map'}" \
    | grep -q "result=0" && echo "saved $out/map.posegraph, map.data" || echo "warning: serialize_map failed" >&2
}

case $action in
  start)
    if running; then echo "already running (pid $(cat "$pidfile")); stop it first" >&2; exit 1; fi
    battery=$(first /battery_state percentage)
    [[ -n $battery ]] || { echo "preflight: base silent (no /battery_state); not starting" >&2; exit 1; }
    pct=$(awk -v p="$battery" 'BEGIN {printf "%d", p * 100}')
    ((pct >= 25)) || { echo "preflight: battery at $pct%; charge it first" >&2; exit 1; }
    docked=$(first /dock_status is_docked)
    [[ $docked == False ]] || { echo "preflight: robot is docked or dock state unknown ('$docked'); undock it first" >&2; exit 1; }
    [[ $(first /kidnap_status is_kidnapped) != True ]] || { echo "preflight: robot is picked up" >&2; exit 1; }
    # Capture first: grep -q would quit early, kill hz with SIGPIPE and fail the pipeline (pipefail).
    scan=$(timeout 15 ros2 topic hz /scan 2>/dev/null | grep -m1 "average rate" || true)
    [[ -n $scan ]] || { echo "preflight: no /scan from the lidar" >&2; exit 1; }
    log=$dir/$(date +%Y%m%d-%H%M%S).log
    # setsid: own session, so the run outlives this SSH connection and can be stopped as a group.
    # A background job starts with Ctrl-C (SIGINT) ignored, and ros2 launch keeps it that way, so
    # restore it: stop relies on it for the explorer to cancel its goal and everything to exit cleanly.
    setsid nohup env --default-signal=INT ros2 launch pet_explorer explore.launch.py "$@" > "$log" 2>&1 < /dev/null &
    echo $! > "$pidfile"
    sleep 5
    if ! running; then echo "the stack exited right away:" >&2; tail -20 "$log" >&2; exit 1; fi
    echo "started (pid $(cat "$pidfile")), battery $pct%, log $log"
    ;;
  status)
    if running; then echo "running (pid $(cat "$pidfile")), log $(latest_log)"; else echo "not running"; fi
    log=$(latest_log)
    [[ -z $log ]] || grep -E '\[explorer\]' "$log" | tail -n 10 || true
    ;;
  logs)
    log=$(latest_log)
    [[ -n $log ]] || { echo "no runs yet"; exit 0; }
    echo "== $log"
    tail -n "${1:-40}" "$log"
    ;;
  save-map)
    save_map
    ;;
  stop)
    running || { echo "not running"; rm -f "$pidfile"; exit 0; }
    pid=$(cat "$pidfile")
    save_map || true
    # Ctrl-C to ros2 launch only: it passes it on, and the explorer cancels its Nav2 goal on the way out.
    kill -INT "$pid"
    for _ in $(seq 1 20); do running || break; sleep 1; done
    if running; then echo "no clean exit after Ctrl-C; terminating" >&2; kill -TERM -- "-$pid"; sleep 3; fi
    if running; then kill -KILL -- "-$pid"; fi
    rm -f "$pidfile"
    echo "stopped"
    ;;
  estop)
    case ${1:-on} in on) value=true ;; off) value=false ;; *) echo "usage: estop [on|off]" >&2; exit 2 ;; esac
    timeout 15 ros2 service call /e_stop irobot_create_msgs/srv/EStop "{e_stop_on: $value}" | tail -1
    ;;
  *)
    echo "usage: ops/field-test.sh start|status|logs|save-map|stop|estop" >&2
    exit 2
    ;;
esac
REMOTE
