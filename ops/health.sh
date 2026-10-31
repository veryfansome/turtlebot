#!/usr/bin/env bash
# Read-only health check of the robot (and the VM, if reachable), run from the Mac:
#   ops/health.sh
# Changes nothing. Lidar and camera are switched off by the robot while it's docked, so those
# are reported as "off (docked)" rather than failures.
set -uo pipefail

ops_dir=$(cd "$(dirname "$0")" && pwd)

echo "== Pi (turtlebot-pi)"
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 turtlebot-pi true 2>/dev/null; then
  echo "  ✗ not reachable over SSH"
  exit 1
fi
ssh -o BatchMode=yes turtlebot-pi bash -s <<'REMOTE'
set +u
source /etc/turtlebot4/setup.bash
ok() { echo "  ✓ $*"; }
bad() { echo "  ✗ $*"; }
off() { echo "  – $*"; }
first() { timeout "$1" ros2 topic echo --once "$2" --field "$3" 2>/dev/null | head -1; }
rate() { timeout 8 ros2 topic hz "$1" 2>/dev/null | grep -m1 -o 'average rate: [0-9.]*' | awk '{printf "%.1f Hz", $3}'; }

echo "  up $(uptime -p | sed 's/^up //'), disk $(df -h / | awk 'NR==2 {print $5}') used"
systemctl is-active --quiet turtlebot4 && ok "turtlebot4 service running" || bad "turtlebot4 service not running"

battery=$(first 8 /battery_state percentage)
if [[ -z $battery ]]; then
  bad "base silent (no /battery_state): restart its application from its web page"
  exit 0
fi
ok "base talking, battery $(awk -v p="$battery" 'BEGIN {printf "%d%%", p * 100}')"
docked=$(first 8 /dock_status is_docked)
[[ $docked == True ]] && ok "docked" || ok "undocked"

for check in "/scan lidar" "/oakd/rgb/preview/image_raw camera"; do
  set -- $check
  hz=$(rate "$1")
  if [[ -n $hz ]]; then ok "$2 $hz"
  elif [[ $docked == True ]]; then off "$2 off (docked)"
  else bad "$2 not publishing ($1)"
  fi
done

# hazard_detection only publishes when something happens, so check for the relay's publisher.
for topic in hazard_detection kidnap_status; do
  ros2 topic info --no-daemon --spin-time 5 /$topic 2>/dev/null | grep -q 'Publisher count: [1-9]' \
    && ok "/$topic relayed" || bad "/$topic not relayed (ops/run.sh pi create3-relays status)"
done
[[ $(first 8 /kidnap_status is_kidnapped) == True ]] && bad "picked up (kidnapped)"

override=$(timeout 15 ros2 service call /_do_not_use/motion_control/get_parameters \
  rcl_interfaces/srv/GetParameters "{names: [safety_override]}" 2>/dev/null | grep -o "string_value='[^']*'" | cut -d"'" -f2)
[[ $override == backup_only ]] && ok "safety_override backup_only" || bad "safety_override is '${override:-unknown}', expected backup_only"
REMOTE

echo "== Ops units on the Pi"
for unit in "$ops_dir"/pi/*.sh; do
  echo "  $("$ops_dir/run.sh" pi "$(basename "$unit" .sh)" status 2>&1)"
done

echo "== VM (turtlebot-vm)"
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 turtlebot-vm true 2>/dev/null; then
  echo "  – not reachable over SSH"
  exit 0
fi
ssh -o BatchMode=yes turtlebot-vm bash -s <<'REMOTE'
cd ~/turtlebot 2>/dev/null || { echo "  ✗ no ~/turtlebot checkout"; exit 0; }
echo "  ✓ ~/turtlebot on $(git branch --show-current) at $(git log --oneline -1)"
[[ -f install/setup.bash ]] && echo "  ✓ workspace built" || echo "  ✗ workspace not built (colcon build --symlink-install)"
bash -ic 'timeout 8 ros2 topic echo --once /battery_state --field percentage' >/dev/null 2>&1 \
  && echo "  ✓ sees the robot over ROS" || echo "  ✗ can't see /battery_state over ROS"
REMOTE
