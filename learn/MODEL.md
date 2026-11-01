# The driving model

One network learns to predict what the robot's sensors will read next (a world model) and, on top of the same learned state, how good a situation is for reaching a goal (values) and what to do (a policy). Predicting every sensor gives dozens of real targets per step, so it learns far more from a little driving than the reward alone would; the reward then only has to teach what's worth doing.

This follows established designs: auxiliary prediction tasks for learning faster from the same data (UNREAL, Jaderberg et al., 2016; SPR, Schwarzer et al., 2020), and a learned world model with the policy and values on top (DreamerV3, Hafner et al., 2023; TD-MPC2, Hansen et al., 2024). Values and the policy are learned offline with IQL (Implicit Q-Learning, Kostrikov et al., 2021), which is built for learning from driving that wasn't optimal and never has to guess the value of actions nobody took.

```
readings now ──► encoder ──► state z ──┬─► dynamics(z, action) ──► next z ──► next z ...
(lidar 60 sectors, floor speed,        │        └─► heads: next lidar, floor speed, wheel speeds,
 wheel speeds, currents, bump, slip,   │                   currents, bump, slip, IR, pose change
 IR, last command)                     │
                                       ├─► value V(z, goal), Q(z, goal, action)   (IQL, reward-trained)
goal relative to the robot ────────────┴─► policy π(action | z, goal)
```

## Inputs, actions, targets

Each step (10 Hz) the encoder sees, scaled to similar ranges:

- the lidar, nearest return in each of 60 sectors, capped at 4 m;
- speed over the floor (optical floor sensor), wheel odometry (forward and turn), wheel speeds, motor currents;
- whether a bump began, wheel slip, wheel stall;
- the 7 front IR proximity sensors, as log(1 + reading): they only see the last 10-15 cm and jump steeply up close (see the [roadmap](../ROADMAP.md)), plus whether they were recorded (sessions from before they were relayed have none);
- the previous drive command (commands take 0.1–0.2 s to show up in the sensors);
- for each sensor (lidar, floor, odometry, wheel speeds, currents, IR), whether its reading is real. A reading is missing when its messages stopped arriving for over 0.5 s (the lidar's go first when Wi-Fi drops big messages) or weren't recorded; missing readings go in as 0 with the flag off.

The action is the drive command: forward speed and turn rate, scaled by the safety layer's caps (0.3 m/s, 1.5 rad/s).

The world model predicts, for each of the next H steps (H = 10, one second), by chaining the dynamics on the commands actually sent: the lidar sectors, floor speed, wheel odometry, wheel speeds, motor currents, whether a bump begins, whether the wheels slip, the IR readings, and how the robot's pose changed (SLAM, in the robot's frame at the start). Predicting a second ahead rather than one step matters: at 10 Hz the next scan is nearly the current one, so a one-step model can get away with copying.

## Training

One loop, one combined loss:

- **World model**: prediction error of every head at every horizon. Each continuous target is standardized (zero mean, unit spread, from the training data) so no head dominates; bump and slip are on/off targets with a classification loss. A sensor's head only counts where its reading is real. A small consistency term keeps the predicted state close to what the encoder makes of the real next readings (as in SPR and TD-MPC2).
- **Values**: IQL's Q and V on top of the state, with goals from hindsight relabeling and the goal-reaching reward ([pet_learn/reward.py](pet_learn/reward.py)). The world model doesn't see goals: physics doesn't depend on where you're going, so it learns from every step.
- **Policy**: advantage-weighted regression (IQL's policy step): it imitates the recorded commands more the better the values say they turned out. Its gradients don't flow into the shared encoder, so the policy can't bend the state to flatter itself. It learns only from hindsight goals (the recorded commands weren't aimed at random goals, so they say nothing about reaching them), counts steps with the goal behind the robot 3 times (turning to face a goal is rare in recorded driving), and learns the commands capped at the safety layer's limits (the controller's turbo goes past them; the world model and values see them as sent). Values and policy are symmetric by construction: each averages its answer for the situation and, flipped back, for its mirror image, so the policy can't come to favour one way of turning (flipping alternate training batches instead left it leaning one way or the other, by chance).

**Missing sensors.** Steps keep going when a sensor's messages stop: the scan is marked missing, and SLAM's last correction carries on with the wheel odometry for the pose. The world model learns from steps up to 60 s after the last scan; goals and rewards, which measure distances on the map, use only steps within 10 s of one (`episodes.scored`), since the pose drifts without scans. A missing motor-current reading carries the last real one forward for the energy cost, and a non-finite reward stops training rather than spreading NaN into the weights. In training, each sensor is also hidden from the encoder at random (5% of windows and transitions each; `--hide`), as if missing, while still being a target: the model learns to predict what it can't see and to drive on what's left. A hidden reading doesn't seed its head's prediction either (heads predict a change from the current reading), so it can't leak back in. The model sees one step at a time, so for now coping means using the other sensors; carrying the last good view through a gap needs memory.

**How long to train.** The world model trains until its val error hasn't improved for 2,000 iterations (`--patience`; at most 50,000), keeping the best: the right length grows with the data. The driver trains a fixed 20,000 iterations: IQL has no held-out loss to stop on. Both train on the Mac's GPU (MPS) when there is one (`--device`).

## Validating on recorded data

The last 20% of each session's scored driving is held out (by time, and of the driving rather than of all its steps: a session can end in a long stretch without scans). Its first half (val) picks the world model's best iteration and when to stop; its second half (test) is what's reported, by both the world model and the driver, so picking can't flatter the result. The driver trains on the same first 80%. The world model is judged against naive baselines on it, per head and horizon: "the next reading equals the current one" for sensors, and "keep the current velocity" for the pose change. Beating those clearly, especially on floor speed and the lidar while turning, means the state has learned real dynamics. Values are checked against the actual discounted returns of held-out episodes. The policy can only really be judged on the robot.

## Code and first results

- [pet_learn/dataset.py](pet_learn/dataset.py): inputs, actions, targets, windows, mirroring. The lidar's 60 sectors are cut in the robot's frame (sector 0 directly behind, straight ahead between sectors 29 and 30), so reversing them mirrors the scan; returns under 15 cm are the robot's own support posts and are ignored.
- [pet_learn/model.py](pet_learn/model.py), [pet_learn/train_world.py](pet_learn/train_world.py): the world model, trained with half of each batch mirrored left to right (physics is symmetric, so this doubles the data honestly). Continuous heads predict the change from the current reading, so "nothing changes" is the default.
- [pet_learn/goals.py](pet_learn/goals.py), [pet_learn/train_driver.py](pet_learn/train_driver.py): goal-relabeled transitions (70% hindsight goals, 30% random poses in mapped free space; unreachable ones skipped) and IQL trained jointly with the world model. Rewards are scaled by 10 for training: one step's reward is a few hundredths, too small for IQL's advantage weights to tell actions apart.

First results, from one 8½-minute session (2026-10-31), on its held-out last 20%:

- World model (128 wide, 10,000 iterations): the robot's own motion is learned well. Error relative to the naive baselines at 1 s: wheel speeds 0.05, odometry 0.27, pose change 0.31. The environment, a little: lidar 0.91, floor speed 0.73, motor currents 0.79. Bumps are anticipated (AUC 0.75-0.9, few held-out bumps so noisy). A wider model overfit within 500 iterations.
- Values: rank correlation 0.72 with the recorded driving's actual returns.
- Policy, goal 2 m away from real held-out states with open space ahead: drives forward fastest toward a goal ahead (0.19 m/s), slows for one behind (0.07) but doesn't turn around, turns right for one on the right, doesn't turn left for one on the left. Not ready for the robot: the recorded driving never turned faster than 0.5 rad/s and has little turning to face a goal. More, more varied driving is the next step.

## Later

- Prediction error is a surprise signal: stage 6 of the [roadmap](../ROADMAP.md) starts from this model; camera features become another head.
- With a dynamics model, the robot can also plan: try candidate command sequences in the model and pick the best by reward and value (TD-MPC2's approach).
