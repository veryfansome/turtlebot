# pet_drive

Driving the TurtleBot 4 without Nav2 (stage 2 of the [roadmap](../../ROADMAP.md)). For now this is the safety layer: the one node that sends drive commands to the wheels, so a scripted wander or a learned policy can drive without being able to hurt the robot or the house.

## The safety layer

Whoever drives publishes the speeds it wants (`geometry_msgs/TwistStamped`: `linear.x` forward speed, `angular.z` turn rate) on `drive/cmd_vel`. The `safety` node applies these rules, in this order, and sends the result to the robot's `cmd_vel` at `rate` (20 Hz):

1. **Picked up** (`kidnap_status`): stop.
2. **Cliff or wheel drop** (`hazard_detection`) reported in the last `hazard_hold`: stop.
3. **Lidar silent**: no `scan` for `scan_timeout`: stop. The lidar has stalled silently on this robot; without it the driver is blind.
4. **No command** for `command_timeout`: stop. Whoever was driving has stopped or crashed.
5. **New bump**: stop for `bump_pause`. While the bumper stays pressed after that, forward speed is set to 0, and backing up and turning go through, so the driver can learn to get itself off things.
6. **Speed caps**: forward speed between `-max_reverse` and `max_forward`, turn rate within `±max_turn`.

The rules are in [pet_drive/safety.py](pet_drive/safety.py), free of ROS and unit-tested; [pet_drive/safety_node.py](pet_drive/safety_node.py) wires them to topics.

When nobody is driving it sends a few zero commands and then goes quiet, so the controller can still drive the robot directly (the controller doesn't go through the safety layer: the person holding it is the safety layer). Don't run it alongside Nav2, which also publishes `cmd_vel`.

It also publishes why it changed the last command on `drive/safety` (`std_msgs/String`: `ok`, `speed capped`, `bumped`, `bumper pressed: no forward`, `lidar silent`, `no command`, `picked up`, `cliff or wheel drop`), for logs and for the recorder: a policy should learn from what the wheels were actually told, not what it asked for.

Hazards come from the base through the `pi/create3-relays` ops unit, which must be applied.

## Running (in the VM)

```bash
ros2 run pet_drive safety
```

Then, to drive slowly forward for as long as this runs (Ctrl-C and it stops within `command_timeout`):

```bash
ros2 topic pub -r 10 /drive/cmd_vel geometry_msgs/msg/TwistStamped "{twist: {linear: {x: 0.1}}}"
```

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `max_forward` | `0.3` | Fastest forward speed (m/s) |
| `max_reverse` | `0.15` | Fastest reverse speed (m/s) |
| `max_turn` | `1.5` | Fastest turn rate (rad/s) |
| `command_timeout` | `0.5` | Stop if no command for this long (s) |
| `scan_timeout` | `1.0` | Stop if no lidar scan for this long (s) |
| `bump_pause` | `0.5` | Stop for this long after a new bump (s) |
| `hazard_hold` | `0.3` | A hazard counts as over once it hasn't been reported for this long (s); the base reports hazards only while they last |
| `rate` | `20.0` | How often commands are sent to the robot (Hz) |

## Not yet checked on the robot

- How often the base reports a held bumper. `hazard_hold` assumes at least every 0.3 s; if it's less often, a held bumper would look like a series of new bumps.
- That reversing works through `cmd_vel`. Nav2's backup didn't reverse on 2026-10-31.
