#!/usr/bin/env bash
# Run on the robot's Pi: turn on relaying of Create 3 topics the VM needs but TurtleBot 4 leaves off.
#   hazard_detection: bumps and other hazards (pet_explorer's bump check)
#   kidnap_status:    the robot was picked up
# Edits the installed republisher config (a backup is kept next to it), so a TurtleBot 4 package
# update can undo it: re-run this afterwards. Safe to run more than once.
#
# Usage: sudo bash enable-create3-relays.sh
set -euo pipefail

config=/opt/ros/jazzy/share/turtlebot4_bringup/config/republisher.yaml
topics=(hazard_detection kidnap_status)

[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
[[ -f $config.orig ]] || cp "$config" "$config.orig"
for topic in "${topics[@]}"; do
  # A disabled entry is the enabled one with "# " in place of its first two spaces of indentation.
  sed -i "s|^      # \"$topic\", |        \"$topic\", |" "$config"
  grep -q "^        \"$topic\", " "$config" || { echo "couldn't enable $topic in $config" >&2; exit 1; }
done
echo "Enabled: ${topics[*]}. Restart the robot's software to apply:"
echo "  sudo systemctl restart turtlebot4"
