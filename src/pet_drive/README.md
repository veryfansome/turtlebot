# pet_drive

Driving the TurtleBot 4 without Nav2 (stage 2 of the [roadmap](../../ROADMAP.md)): the safety layer, the one node that sends drive commands to the wheels, so a scripted wander or a learned policy can drive without being able to hurt the robot or the house; and the recorder, which collects the driving sessions a policy learns from.

## The safety layer

Whoever drives publishes the speeds it wants (`geometry_msgs/TwistStamped`: `linear.x` forward speed, `angular.z` turn rate) on `drive/cmd_vel`. The `safety` node applies these rules, in this order, and sends the result to the robot's `cmd_vel` at `rate` (20 Hz):

1. **Picked up** (`kidnap_status`): stop.
2. **Cliff or wheel drop** (`hazard_detection`) reported in the last `hazard_hold`: stop.
3. **Lidar silent**: no `scan` for `scan_timeout`: stop. The lidar has stalled silently on this robot; without it the driver is blind.
4. **No command** for `command_timeout`: stop. Whoever was driving has stopped or crashed.
5. **Speed caps**: forward speed between `-max_reverse` and `max_forward`, turn rate within `±max_turn`.

Bumps aren't a rule: the robot is built to bump into things, and the driver should learn from them, not be stopped by them. Bumps, and pushing without moving, are for the recorder and the reward.

The rules are in [pet_drive/safety.py](pet_drive/safety.py), free of ROS and unit-tested; [pet_drive/safety_node.py](pet_drive/safety_node.py) wires them to topics.

When nobody is driving it sends a few zero commands and then goes quiet, so the controller can still drive the robot directly (the controller doesn't go through the safety layer: the person holding it is the safety layer). Don't run it alongside Nav2, which also publishes `cmd_vel`.

It also publishes why it changed the last command on `drive/safety` (`std_msgs/String`: `ok`, `speed capped`, `lidar silent`, `no command`, `picked up`, `cliff or wheel drop`), for logs and for the recorder: a policy should learn from what the wheels were actually told, not what it asked for.

When it stops the robot for a reason, it tells whoever is nearby (`feedback`, on by default) with tones on the base's speaker (`cmd_audio`; it can't speak) and a colour on its light ring (`cmd_lightring`), held while stopped:

| Stop | Sound | Light ring |
|---|---|---|
| No command (the driver stopped) | one short beep | amber, briefly |
| Lidar silent | three falling tones | red |
| Cliff or wheel drop | fast high warble | magenta |
| Picked up | three rising tones | blue |
| Driving again after one of the last three | two rising blips | back to normal |

The tones and colours are in [pet_drive/feedback.py](pet_drive/feedback.py). Nothing plays while nobody is driving, except two things: one long note once when it starts and the base is listening (so you know a recording has begun), and two low beeps when you press the controller's rescue button (Y, just before freeing a stuck robot, see [Recording driving sessions](#recording-driving-sessions)), so you know the press was seen.

Hazards come from the base through the `pi/create3-relays` ops unit, which must be applied.

## Recording driving sessions

`launch/record.launch.py` starts SLAM, the safety layer and `ros2 bag record` (MCAP) into a new directory under `~/drives/`. It drives nothing itself: drive with the controller, or have a driver publish on `drive/cmd_vel`. From the Mac:

```bash
ops/field-test.sh start record    # preflight checks, then record in the background
ops/field-test.sh stop            # save the map, close the bag
ops/pull-drives.sh                # copy the bags to data/drives/ on the Mac
```

What's recorded, and why:

| Topic | For |
|---|---|
| `/scan` | What the driver sees |
| `/odom` | What the wheels did |
| `/mouse` | The base's optical floor sensor: what the robot actually did over the floor. Wheels turning while this says it isn't moving is pushing or slipping |
| `/imu` | Jolts from collisions the bumper misses |
| `/wheel_status` | Motor currents, which rise when it pushes against something |
| `/wheel_vels`, `/slip_status` | Actual wheel speeds, and the base's own wheel-slip detection |
| `/ir_intensity` | The 7 front infrared proximity sensors, at bumper height: things below the lidar's scan plane, like chair legs or toys |
| `/hazard_detection` | Bumps (only at the moment of contact), cliffs, wheel drops |
| `/kidnap_status`, `/dock_status`, `/battery_state` | Picked up, docked, battery |
| `/tf`, `/tf_static`, `/map` | Where it is: SLAM's `map` → `odom` is movement measured by SLAM, not the wheels |
| `/cmd_vel` | What the wheels were told, by the safety layer or the controller |
| `/drive/cmd_vel`, `/drive/safety` | What a driver asked for, and why the safety layer changed it |
| `/joy` | The controller: whether a person was driving, and rescues (press Y just before freeing a stuck robot) |

`camera:=true` adds the camera's preview. It's off by default: the robot only offers raw frames, about 5-6 MB/s over Wi-Fi at 30 Hz, which risks losing lidar scans. Compressed frames would need a camera change on the Pi.

## Running the safety layer alone (in the VM)

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
| `hazard_hold` | `0.3` | A cliff or wheel drop counts as over once it hasn't been reported for this long (s) |
| `rate` | `20.0` | How often commands are sent to the robot (Hz) |
| `feedback` | `true` | Play tones and colour the light ring when it stops the robot |

## Checked on the robot (2026-10-31)

- Commands go out at 20 Hz while driving, and stop within `command_timeout` when the driver stops.
- Reversing works: 2 s at 0.1 m/s through the safety layer moved the robot 23 cm straight back.
- The base reports a bump only at the moment of contact (two reports at once), not while the bumper stays pressed. Pushing against something afterwards shows up only as commanded speed without matching motion.
- Tones and light ring colours sent from the VM reach the base through the relay (the lidar-silent pattern and red ring, heard and seen).

Not yet checked: whether the ring keeps our colour for a long stop, or the robot takes it back after a while.
