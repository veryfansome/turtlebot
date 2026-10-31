# Ops: changing the robot and the VM safely

The robot is expensive and easy to break, so every change to the Pi, the Create 3 base or the VM goes through a script in this folder, never through hand-typed commands. The repo is then the record of what has been changed on each machine, and every change can be undone.

Run everything from the Mac, in the repo root. It reaches the machines over SSH as `turtlebot-pi` and `turtlebot-vm` (see [vm/README.md](../vm/README.md) for the key and the `~/.ssh/config` entries).

## Health check

```bash
ops/health.sh
```

Read-only. Checks the robot's software, base, battery and dock state, lidar, camera, relays and `safety_override`, the state of each unit, and whether the VM can see the robot over ROS. Lidar and camera show as "off (docked)" while the robot is docked, which is normal.

## The VM and field tests

```bash
ops/vm-sync.sh [branch]        # pull, rebuild and test ~/turtlebot in the VM (pushed commits only)
ops/field-test.sh start        # preflight checks, then run SLAM + Nav2 + explorer in the background: drives the robot
ops/field-test.sh status       # running or not, and the latest explorer lines
ops/field-test.sh logs [lines] # the end of the run's log (kept in ~/field-tests/ on the VM)
ops/field-test.sh save-map     # save the SLAM map to ~/maps/<run>/ on the VM
ops/field-test.sh stop         # save the map, then stop the run; the explorer cancels its goal first
ops/field-test.sh estop on|off # the base's e-stop: cut or restore drive motor power
```

`start` refuses if the base is silent, the battery is under 25%, the robot is docked or picked up, or the lidar isn't publishing. Only start a run with someone watching the robot.

## Units

A unit is one reversible change, a script under `ops/pi/` or `ops/vm/`, run with:

```bash
ops/run.sh <pi|vm> <unit> status   # applied, not applied, or partly applied
ops/run.sh <pi|vm> <unit> diff     # what apply would change; changes nothing
ops/run.sh <pi|vm> <unit> apply    # make the change, restart what's needed, verify
ops/run.sh <pi|vm> <unit> revert   # undo it, restart, verify
```

- `status` and `diff` are read-only. `apply` and `revert` need sudo on the target, unless there's nothing to do: in a terminal you're asked for the password; without one they stop and print the command to run yourself.
- `apply` and `revert` are safe to repeat: each is a no-op when already done.
- Before changing a file, a unit backs it up on the target under `/var/lib/turtlebot-ops/backups/<unit>/`: `<file>.orig` the first time (never overwritten), plus a timestamped copy every time. Every change is logged in `/var/lib/turtlebot-ops/changes.log`.
- After a change the unit verifies the machine (e.g. the relayed topics come back). If that fails, it puts the previous file back and restarts again.
- Shared logic is in [lib.sh](lib.sh); a new unit defines the file it changes, the apply and revert transforms, what to restart, and how to verify.

| Unit | Changes | Why |
|---|---|---|
| `pi/create3-relays` | Uncomments `hazard_detection` and `kidnap_status` in `/opt/ros/jazzy/share/turtlebot4_bringup/config/republisher.yaml`; restarts `turtlebot4` | The explorer's bump check needs hazards; pickup handling needs kidnap status. A TurtleBot 4 package update can overwrite the file: check `status` after updates. |

## Not scripted

Some changes are risky enough that they stay manual, done by a person with the steps written down:

- Pi network or Wi-Fi settings: the easiest way to lose access to the robot.
- Firmware updates for the Pi or the Create 3 base, and `apt upgrade` on the Pi.
- Disk partitions or the boot configuration.
- The Create 3 base's own config file, in its web page (Application → Configuration). Currently set there: `safety_override: "backup_only"` under `/**/motion_control:`, so the robot can reverse more than about an inch. The key needs the `/**/` prefix to match the base's `/_do_not_use` namespace. `ops/health.sh` checks the live value.
