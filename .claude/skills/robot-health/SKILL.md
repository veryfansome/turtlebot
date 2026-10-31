---
name: robot-health
description: Read-only health check of the TurtleBot 4 (Pi, Create 3 base, lidar, camera, relays, safety_override) and the VM. Use before and after any change to the robot or VM, when the robot isn't behaving (won't drive, no camera, no scan, explorer stuck), or when asked how the robot is doing.
---

# Robot health check

Run from the repo root on the Mac:

```bash
ops/health.sh
```

It only reads: safe to run any time, without asking. It takes 1–2 minutes.

## Reading the result

- `–` lines are expected states, not failures. Lidar and camera show "off (docked)": the robot's power saver turns them off on the dock.
- **Base silent** (no `/battery_state`): the Create 3's software isn't running or is still starting (it can take about 2.5 minutes after a restart). Don't restart anything yourself. Tell the user. Restarting the base's application from its web page fixed it before.
- **Camera not publishing while undocked**: a known problem. After the camera is stopped (e.g. docking), its start request sometimes never gets answered. Confirm in the journal: `ssh turtlebot-pi 'journalctl -u turtlebot4 --since "-10 min" | grep -E "\[oakd\]|OAKD"'` shows `start_camera ... sending request` with no `Starting camera.` after it. The fix is `sudo systemctl restart turtlebot4` on the Pi, which needs the user's sudo password: ask them to run it.
- **Relay not relayed** or **safety_override wrong**: see `ops/README.md`. Relays are a unit (use the push-config skill); `safety_override` lives in the base's web config, which only the user changes.
- **VM not reachable**: its address can change (it comes from the router). Ask the user for `ip -4 addr show` from the VM and update `HostName` under `Host turtlebot-vm` in `~/.ssh/config`.

## Rules

- Report what the check shows. Don't fix anything as part of a health check.
- For ROS commands on the Pi beyond this script, first `source /etc/turtlebot4/setup.bash`. Use `ros2 topic echo --once` or `ros2 topic hz` to see whether data flows. If you need a publisher count, pass `--no-daemon --spin-time 5`: the default discovery wait gave 0 for topics that had publishers.
