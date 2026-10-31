---
name: field-test
description: Run the explorer stack (SLAM, Nav2, pet_explorer) on the VM so the TurtleBot 4 drives around, watch it, and stop it, through ops/field-test.sh. Use when the user wants to try code on the robot. Starting it moves the robot, so it needs the user's explicit go in the current conversation; stopping and the e-stop never need permission.
---

# Field test

`ops/field-test.sh` controls a background run on the VM. Logs are in `~/field-tests/` on the VM, one per run.

## Before starting

1. `ops/vm-sync.sh`, so the VM runs the code you mean to test, with tests passing.
2. `ops/health.sh` (robot-health skill). Don't start if the base is silent, the lidar isn't publishing, or `safety_override` is wrong. The camera isn't needed by the explorer yet, so a stuck camera alone doesn't block a run.
3. **Ask the user for a go,** and say what will run (branch and commit, launch arguments). Their go means they're watching the robot, stairs and drop-offs are blocked, cables and fragile things are out of the way, and the robot is undocked. A go is for one run: ask again before restarting it, even if they approved the last one.

## Running

```bash
ops/field-test.sh start [launch args]   # e.g. rviz:=true doesn't work over SSH; the user opens RViz in the VM
ops/field-test.sh status                # running or not + the last explorer lines
ops/field-test.sh logs [lines]          # the end of the full log (SLAM, Nav2, explorer)
```

- `start` checks first and refuses if the base is silent, the battery is under 25%, the robot is docked or picked up, or there's no lidar data.
- While it runs, check `status` every minute or so and tell the user what the explorer is doing: frontiers reached, goals abandoned and why, stuck and probing.
- For RViz, the user runs `ros2 launch turtlebot4_viz view_navigation.launch.py` in a terminal in the VM's own desktop.

## Stopping

```bash
ops/field-test.sh stop          # normal stop: the explorer cancels its goal, then everything exits
ops/field-test.sh estop on      # emergency: cuts the drive motors' power at the base
ops/field-test.sh estop off     # restore motor power (only when the user says it's safe)
```

- Stop without asking when the user says stop, when something looks unsafe (repeated bumps, kidnapped, battery under 20%), when the explorer reports `exploration complete`, or when you're about to end your turn without the user watching.
- Use `estop on` when a normal stop isn't enough or the robot is doing something dangerous; then `stop` as well. Never turn the e-stop off without the user's say-so.
- Don't leave a run going silently: if it's still running when you reply, say so.

## After

Summarize the run from `ops/field-test.sh logs` and `status`: duration, frontiers reached, failures and their causes, anything new. Then `ops/health.sh` again. The camera in particular often needs a `turtlebot4` restart after docking and undocking.
