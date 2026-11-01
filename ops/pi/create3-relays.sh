#!/usr/bin/env bash
# Relay Create 3 base topics through the Pi that TurtleBot 4 ships commented out:
#   hazard_detection: bumps, cliffs, wheel drops (the safety layer, recordings)
#   kidnap_status:    the robot was picked up
#   mouse:            the optical floor sensor: actual motion over the floor (recordings)
#   slip_status:      the base's own wheel-slip detection (recordings)
#   wheel_vels:       actual wheel speeds (recordings)
#   ir_intensity:     the 7 front infrared proximity sensors, at bumper height (recordings)
# Edits the installed republisher config, so a TurtleBot 4 package update can undo it; check
# `status` after updates.
source "$(dirname "$0")/../lib.sh"

UNIT=create3-relays
TARGET_FILE=/opt/ros/jazzy/share/turtlebot4_bringup/config/republisher.yaml
TOPICS=(hazard_detection kidnap_status mouse slip_status wheel_vels ir_intensity)

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
