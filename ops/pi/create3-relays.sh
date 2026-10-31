#!/usr/bin/env bash
# Relay two Create 3 base topics through the Pi that TurtleBot 4 ships commented out:
#   hazard_detection: bumps and other hazards (pet_explorer's bump check)
#   kidnap_status:    the robot was picked up
# Edits the installed republisher config, so a TurtleBot 4 package update can undo it; check
# `status` after updates.
source "$(dirname "$0")/../lib.sh"

UNIT=create3-relays
TARGET_FILE=/opt/ros/jazzy/share/turtlebot4_bringup/config/republisher.yaml
TOPICS=(hazard_detection kidnap_status)

# A disabled entry is the enabled one with "# " in place of the first two spaces of its indent.
unit_apply_transform() {
  local args=() t
  for t in "${TOPICS[@]}"; do args+=(-e "s|^      # \"$t\", |        \"$t\", |"); done
  sed "${args[@]}"
}
unit_revert_transform() {
  local args=() t
  for t in "${TOPICS[@]}"; do args+=(-e "s|^        \"$t\", |      # \"$t\", |"); done
  sed "${args[@]}"
}

unit_after_change() { systemctl restart turtlebot4; }

has_publisher() { ros2 topic info --no-daemon --spin-time 5 "$1" 2>/dev/null | grep -q 'Publisher count: [1-9]'; }

unit_verify_apply() {
  ros_env
  local t
  for t in "${TOPICS[@]}"; do
    ops_wait_for "/$t is relayed" 120 has_publisher "/$t" || return 1
  done
}

unit_verify_revert() {
  ops_wait_for "turtlebot4 service running" 60 systemctl is-active --quiet turtlebot4
}

ops_main "$@"
