---
name: field-test
description: Run a stack on the VM through ops/field-test.sh, watch it, and stop it - record (SLAM + safety layer + a bag of a driving session, for learning to drive) or explore (SLAM + Nav2 + pet_explorer, paused). Use when the user wants to record driving or try code on the robot. Starting needs the user's explicit go in the current conversation; stopping and the e-stop never need permission.
---

# Field test

`ops/field-test.sh` controls a background run on the VM. Logs are in `~/field-tests/` on the VM, one per run. Two modes:

- `record`: SLAM, the safety layer and `ros2 bag record` into `~/drives/<time>/` (see [src/pet_drive](../../../src/pet_drive/README.md)). It drives nothing itself: the user drives with the controller, or a driver publishes on `drive/cmd_vel`. This is how training data gets collected.
- `explore`: SLAM, Nav2 and the frontier explorer, which drives the robot. Paused since 2026-10-31 in favor of the learned driver.

## Before starting

1. `ops/vm-sync.sh`, so the VM runs the code you mean to test, with tests passing.
2. `ops/health.sh` (robot-health skill). Don't start if the base is silent, the lidar isn't publishing, or `safety_override` is wrong. The camera isn't needed by the explorer yet, so a stuck camera alone doesn't block a run.
3. **Ask the user for a go,** and say what will run (mode, branch and commit, launch arguments). Their go means they're watching the robot, stairs and drop-offs are blocked, cables and fragile things are out of the way, and the robot is undocked. A go is for one run: ask again before restarting it, even if they approved the last one.

## Running

```bash
ops/field-test.sh start record [args]   # e.g. camera:=true (raw frames, heavy on Wi-Fi)
ops/field-test.sh start explore [args]  # rviz:=true doesn't work over SSH; the user opens RViz in the VM
ops/field-test.sh status                # running or not, the last explorer/safety lines, the bag's size
ops/field-test.sh logs [lines]          # the end of the full log
ops/pull-drives.sh                      # after recording: copy bags to data/drives/ on the Mac
```

- `start` checks first and refuses if the base is silent, the battery is under 25%, the robot is docked or picked up, or there's no lidar data.
- While it runs, check `status` every minute or so. Exploring: tell the user what the explorer is doing (frontiers reached, goals abandoned and why, stuck and probing). Recording: that the bag is growing, and any safety-layer stops.
- For RViz, the user runs `ros2 launch turtlebot4_viz view_navigation.launch.py` in a terminal in the VM's own desktop.

## Stopping

```bash
ops/field-test.sh save-map      # save the SLAM map to ~/maps/<run>/ (stop does this too)
ops/field-test.sh stop          # normal stop: saves the map, the explorer cancels its goal, then everything exits
ops/field-test.sh estop on      # emergency: cuts the drive motors' power at the base
ops/field-test.sh estop off     # restore motor power (only when the user says it's safe)
```

- Stop without asking when the user says stop, when something looks unsafe (repeated bumps, kidnapped, battery under 20%), when the explorer reports `exploration complete`, or when you're about to end your turn without the user watching.
- Use `estop on` when a normal stop isn't enough or the robot is doing something dangerous; then `stop` as well. Never turn the e-stop off without the user's say-so.
- Don't leave a run going silently: if it's still running when you reply, say so.
- If the user rejects a `stop` or `estop` you sent, it may already have run: check `ops/field-test.sh status` right away and tell them what state the run is in.
- A normal stop logs launch's Ctrl-C handling and the explorer's `Cancelling the current goal`. If the log ends without them, or `stop` printed "no clean exit after Ctrl-C", the clean shutdown didn't happen: say so.

## Pickup tests

To test pickup handling, the user lifts the robot mid-run and sets it down elsewhere. Watch `status` for `Picked up`, `SLAM paused`, `Saved SLAM state`, `Set down`, `SLAM reset`, `SLAM resumed`, `Turned a full circle`. Ask the user where they put it down relative to where they picked it up (e.g. "2 m toward the kitchen, turned to face the window"): that's the right answer the map matcher will be tested against, so note it next to the saved maps (`~/maps/explorer-<time>/` on the VM).

## After

Summarize the run from `ops/field-test.sh logs` and `status`: duration, frontiers reached, failures and their causes, anything new. Then `ops/health.sh` again. The camera in particular often needs a `turtlebot4` restart after docking and undocking.
