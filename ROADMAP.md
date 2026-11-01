# Roadmap: a TurtleBot 4 that explores the house on its own

The goal is a robotic pet: a TurtleBot 4 that moves around the house on its own initiative, notices things, and gets more interesting as it learns. This is a fun ML project, built up in stages, with a working robot at the end of each one.

## Design decisions

**We drive the wheels ourselves, and the driving is learned.** Nav2 is paused (2026-10-31). Three days of field runs showed its limits in a cluttered home: it only avoids what's on its costmap, so soft cushions, thin chair legs, a floor strip and a couch edge left it pushing in place, and its fallback is to give up on a goal and try another. Tuning helped and each run found the next problem (see [Lessons from Nav2](#lessons-from-nav2)). Instead, a policy will drive the wheels directly from what the robot senses, rewarded for reaching a goal efficiently, so it can learn things a planner never will: backing off a pressed bumper, wiggling off a strip, turning away from a cushion. Every command it sends goes through a hand-written safety layer it can't override.

**Driving is a skill; where to go is a separate choice.** The driver learns one thing: get from where it is to a given pose (position and heading) as efficiently as possible. Choosing goals (frontiers, novelty, curiosity, the dock) is the level above it, and that's where the pet's personality lives. Nav2 is the baseline: same start, same goal, Nav2 vs the learned driver.

**The "brain" runs off the robot.** The robot's Raspberry Pi 4 is busy running the robot's own software. ML training runs on the Mac (PyTorch on the Apple Silicon GPU), ROS 2 and the policy run in the VM (see [vm/README.md](vm/README.md)), and both talk to the robot over Wi-Fi. Anything streamed should use compressed image topics. Wi-Fi adds 0.1–0.2 s of lag, so the policy is trained with that lag, and the safety layer doesn't depend on fresh data from the VM to stop.

**Two kinds of novelty, kept separate.** "I haven't been / seen this here" (coverage novelty) and "I didn't expect that" (surprise) are different signals. Keeping them apart makes it possible to balance them later.

**Every change to the robot or VM is a reversible script.** The robot is expensive and easy to break, so changes go through [ops/](ops/README.md): units with status, diff, apply and revert, backups and automatic rollback, plus a health check. Network, firmware and the base's own config stay manual.

## Stages

### 1. Map the house and keep the map current

- [x] Build a map with SLAM, following the [Generating a map](https://turtlebot.github.io/turtlebot4-user-manual/tutorials/generate_map.html) tutorial. SLAM (slam_toolbox) stays: the map is used for rewards and for knowing where the robot has been.
- [x] Save the map at the end of a run (`ops/field-test.sh stop` saves it to `~/maps/` on the VM, both as an image and as slam_toolbox's full state).
- [ ] Localize on a saved map (slam_toolbox's localization mode) and keep the map up to date as furniture moves, e.g. by reloading a saved map and continuing to map onto it.
- [ ] **Recover from being picked up.** SLAM tracks the robot mostly by wheel motion, so carrying it somewhere else corrupts the map. The plan:
  - [x] On pickup (`kidnap_status`, relayed from the base), stop, pause SLAM (`pause_new_measurements`) and save its state (`serialize_map`) and the grid. Built in the explorer; worked on the robot (2026-10-31). Moves to the driver.
  - [x] When set down, reset SLAM (`reset`; it doesn't resume a paused SLAM by itself, so it's resumed explicitly), turn a full circle, and start a fresh map. Worked on the robot (2026-10-31).
  - Repeatedly match the fresh map against the saved one (every rotation and offset, with FFT cross-correlation), keeping every placement that fits well as a hypothesis and re-scoring only those as the fresh map grows.
  - Accept a match only when one hypothesis clearly wins (e.g. above 95%) and fits well in absolute terms; then reload the saved map at that pose (`deserialize_map`). If nothing fits, the robot is somewhere new: keep the fresh map. A false match corrupts the saved map, so when in doubt, don't match.
  - While several hypotheses remain, drive to where they predict the most different views (active localization), so one look settles it.

### 2. Learn to drive

- [ ] **Safety layer** ([src/pet_drive](src/pet_drive/README.md)): the only way commands reach the wheels. Caps speeds, and stops when commands stop coming, the lidar goes quiet, the robot is picked up, or a wheel drops or a cliff is sensed. Bumps aren't blocked: the robot is built to bump, and the driver learns from them. Built and checked on the robot (2026-10-31): commands, the deadman and reversing work. The base reports a bump only at the moment of contact, so pushing afterwards shows up only as commanded speed without matching motion, which the recorder and reward need to compute.
- [ ] **Recorder**: `ops/field-test.sh start record` bags the lidar, wheel odometry, the base's optical floor sensor (`/mouse`), IMU, motor currents, hazards, the SLAM pose and every drive command, whoever is driving; `ops/pull-drives.sh` copies the bags to the Mac. Commanded vs actual motion is the "wheels spinning but not moving" signal. First session recorded (2026-10-31: 8½ minutes driving with the controller, 72 bumps). [learn/](learn/README.md) turns sessions into 10 Hz training steps on the Mac, without ROS.
- [ ] **The task and its reward.** An episode is one goal pose in mapped free space: it succeeds when the robot is within about 20 cm and facing the goal's heading (within about 20 degrees), and fails on a timeout or a rescue (being picked up). The policy sees the goal's distance and direction, and the heading to end at, relative to itself. Reward, every step:
  - **Progress**: the drop in distance to the goal along the shortest path through the known map (not the straight line, which would reward pushing into the couch to reach a goal behind it); once close, the drop in heading error.
  - **Arrival**: a fixed bonus for reaching the goal pose.
  - **Time**: a fixed penalty per step. Distance isn't a separate term: with speeds capped, a longer path already costs more time.
  - **Energy**: a penalty on motor current over time (the battery percentage is far too coarse to use). Pushing against a cushion then costs energy and time and earns no progress, with no special rule for it.
  - **Rescue**: needing a person to free it ends the episode as a failure, with a penalty as costly as 30 s of driving, whatever the goal. Press Y on the controller just before freeing it (lifting it out, or freeing it by hand); the steps until it's set down (or 10 s) are left out of training. Recordings include a few rescues on purpose (e.g. driving head-on into the floor strip), so the model learns what leads to them.
  - Bumps and pushing without moving have no terms of their own: they're paid for in time and energy. Energy costs as much per amp as time per second: about 20% on top of the time cost while driving normally (0.19 A), 40-70% while pushing (0.42 A).
  - Built in [learn/](learn/README.md) (shortest-path distance, reward, hindsight episodes) and checked on the first session: reward tracks SPL (correlation 0.73), direct episodes earn about +0.8, meandering ones down to about −3.
- [x] **Session report** (`pet_learn.report`): per episode and session, success rate, SPL (success weighted by the shortest path over the path driven: higher is better, 1.0 is perfect), time and energy per metre of shortest path, rescues, bumps, time stuck, so the reward can be checked against what actually happened.
- [ ] **Nav2 as the baseline**: run Nav2 ([config/nav2.yaml](config/nav2.yaml)) and the learned driver from the same starts to the same goals, including goals past the known trouble spots, and compare their reports.
- [ ] **Front IR proximity sensors**: the base has 7 infrared sensors across the front at bumper height (`ir_intensity`), which should see things the lidar's scan plane misses, like chair legs, a cushion, maybe the floor strip. Relayed (`pi/create3-relays`), recorded, in the training steps, and an input and a head of the driving model ([learn/MODEL.md](learn/MODEL.md)); next, sessions recorded with them, to see what the model makes of them. Checked on the robot (2026-10-31): they point at 75°, 50°, 25° left, straight ahead, 25°, 50°, 75° right (the one named `front_center_left` is the centre one), each in a narrow cone. They only see things up close: nothing at 30 cm, a faint reading at 15 cm, strong at 5 cm. Straight ahead (centre sensor, nothing there reads 0-5): hand 8 at 15 cm, 500 at 5 cm; dark chair leg 8 at 15 cm, 180 at 5 cm; light toy 32 at 15 cm, 485 at 5 cm. The two 75° sensors read weaker than the front ones (65-80 for a hand at 5 cm). So the reading depends on the object as much as on distance, and the useful range is the last ~10 cm: half a second's warning at full speed, more when slow. They report at about 62 Hz; on the dock the centre ones read about 20.
- [ ] **Demonstrations**: drive with the controller through the places Nav2 got stuck (chair legs, the cushion, the floor strip, the couch edge), 20–30 minutes to start. Recordings don't need goals: any later pose in a session can be the goal for the steps before it (hindsight relabeling), so one session splits into many episodes of any length, each scored against its own goal.
- [ ] **A driving model, bootstrapped from recorded driving** ([learn/MODEL.md](learn/MODEL.md)): one network with a shared state, trained on several objectives at once:
  - a world model: from the current readings and the drive command, predict every sensor up to a second ahead (lidar, floor speed, wheel speeds and currents, bumps, slip, pose change). Dozens of real targets per step, so it learns a lot from a little driving, and it doesn't need goals;
  - values and a policy for reaching goal poses (IQL), on the same state, from recorded sessions relabeled with goals and scored with the reward. Not copying the driving: the reward decides what was good, so meandering stretches still teach.
  - Validated offline on held-out driving: the world model against "nothing changes" baselines, values against actual returns.
  - Built ([learn/MODEL.md](learn/MODEL.md)); first results from one session: own motion learned well, the environment a little, values track returns (rank correlation 0.72), the policy only partly goal-directed. Needs more, more varied driving.
- [ ] **Reinforcement learning on the robot**: off-policy (e.g. SAC), learning from everything ever recorded, relabeled with goals, so every minute of driving counts. Training goals are reachable poses in the map, near ones first, then farther, plus goals past the known trouble spots.
- [ ] Maybe: pretrain in the TurtleBot 4 Gazebo simulator for walls and open space. It can't model a cushion or a floor strip, so the house stays the main teacher.
- Watch: Wi-Fi lag, the lidar stalling (see Known issues), and about an hour of battery per session, so data collection happens in short sessions.

### 3. Explore on its own

- [ ] Drive toward frontiers (edges between mapped and unmapped space) with the learned driver: the frontier code in [src/pet_explorer](src/pet_explorer/README.md) carries over as the goal picker. Coverage novelty ("haven't been here lately") picks goals too.
- [ ] Measure progress by how much of the map gets filled in.
- [ ] While exploring, record camera, position and map data to rosbags, so stages 5 and 6 have data to work with.

### 4. A simple pet routine

- [ ] A small state machine or behavior tree: explore → wander → return to the dock when the battery is low → rest → repeat. Docking can use the TurtleBot 4's own dock action.
- [ ] This is the framework later stages plug into. The robot's power saver turns the camera off while docked, so resting means not seeing.

### 5. Look at things and remember them

- [ ] Turn camera frames into feature vectors with a pretrained vision model (e.g. DINOv2 or CLIP) on the Mac.
- [ ] Store them by location on the map.
- [ ] Coverage novelty: a view is new if it's unlike anything seen before at that spot. This also catches changes ("something moved in the kitchen").

### 6. A world model

- [ ] The driving model (stage 2) is already a world model for the robot's own senses; this stage adds the camera.
- [ ] Train a model to predict the feature vector of the next view from the current view and the planned move. Predicting features rather than raw pixels is much easier to train, and it uses the data from stage 5.
- [ ] Its prediction error is the measure of surprise.

### 7. Curiosity as a reward

- [ ] Reward choosing goals where the world model is surprised.
- [ ] Watch for the "noisy TV" problem: raw prediction error gets hooked on things that are just random (a TV, a window, a moving fan). The usual fix is to reward learning progress (surprise that shrinks as the model learns) rather than surprise itself. That also gives the pet-like behavior: drawn to new things, bored once it understands them.
- [ ] Getting its bearings after being picked up (stage 1) is the same idea in a simple form: go where you're most uncertain about what you'll see.

## Lessons from Nav2

The frontier explorer drove Nav2 from 2026-10-29 to 2026-10-31. Its first full run mapped 90% or more of the kitchen and living room in about 4½ minutes, then later runs got stuck more and more. What we learned, kept for the driver and the safety layer:

- **Costmaps were too cautious, then too sticky.** The TurtleBot 4 settings padded every obstacle so narrow passages looked blocked; [config/nav2.yaml](config/nav2.yaml) narrows them. Obstacles stay marked until a lidar beam passes through them, so a person who lingered, or what the tilted lidar saw while the robot was carried, left phantom walls.
- **The collision monitor ate commands.** With `base_shift_correction` on, it needs odometry at each scan's exact time; odometry arrives over Wi-Fi a little later, so it stopped the robot 48 times in one run. Turning it off fixed that.
- **Invisible obstacles.** Soft cushions don't press the bumper; thin chair legs give the lidar only a few points; a floor strip is under the scan line; a couch edge can be over it. The robot pushed against them for up to a minute.
- **A held bumper counted once.** Pressed against a chair leg, the bumper never released, so "give up after 2 bumps" never fired.
- **Nav2's backup didn't reverse** (2/2 tries on 2026-10-31, robot not pinned), though the robot reverses fine when commanded directly; not resolved.

## Known issues

- **The lidar stalls silently.** Twice on 2026-10-31 it stopped publishing with no error while still spinning, once right as the robot got caught on the floor strip. `sudo systemctl restart turtlebot4` on the Pi fixes it. The safety layer stops the robot when it happens.
- **The Pi's Wi-Fi can stop passing large packets.** On 2026-10-31, after hundreds of `brcmfmac` driver errors, packets that need splitting (a lidar scan does) were lost while small ones got through, so the VM saw odometry but no scans. A Pi reboot fixed it; `ops/health.sh` checks for it.
- **Camera doesn't restart after docking.** The robot's power saver stops the camera on the dock; on undocking, the start request sometimes never gets answered. `sudo systemctl restart turtlebot4` on the Pi fixes it. Turning the power saver off (as an ops unit) would avoid it.
- **Low obstacles are invisible to the lidar.** It scans one plane, so things lower than that (e.g. the floor strip) only show up as bumps. The base's front IR proximity sensors are the cheaper first try (stage 2). The camera's depth data could fill this in too, but the robot's camera setup publishes only raw RGB preview frames (no depth, no compressed images): about 5-6 MB/s over Wi-Fi. Turning on compressed frames and depth on the Pi (as an ops unit) comes before recording camera data routinely.
- **Battery.** Off the dock with the lidar and camera on, the battery went from 58% to 28% in about 50 minutes (2026-10-31); field tests refuse to start below 25%.

## Background reading

- Pathak et al., 2017, *Curiosity-driven Exploration by Self-supervised Prediction* (ICM).
- Burda et al., 2018, *Exploration by Random Network Distillation* (RND).
- Schmidhuber's work on curiosity and learning progress.
- Burgard, Fox and Thrun's work on active Markov localization (late 1990s): choosing moves that best tell apart where the robot might be.
- Haarnoja et al., 2018, *Soft Actor-Critic*; and work on learning to drive and walk directly on real robots (e.g. Smith et al., 2022, *A Walk in the Park*; Wu et al., 2022, *DayDreamer*).

## Not yet

- Audio: the TurtleBot 4 has no microphone. Adding one would mean a USB microphone on the Pi.
