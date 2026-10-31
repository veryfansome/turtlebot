# Roadmap: a TurtleBot 4 that explores the house on its own

The goal is a robotic pet: a TurtleBot 4 that moves around the house on its own initiative, notices things, and gets more interesting as it learns. This is a fun ML project, built up in stages, with a working robot at the end of each one.

## Design decisions

**Nav2 handles movement for now; the learning decides where to go.** Nav2 plans paths, avoids obstacles and docks, and the learned part picks goals ("go look over there", "go back to that spot", "rest on the dock"). TurtleBot 4's Nav2 settings were too cautious for a cluttered home, so we run our own copy, [config/nav2.yaml](config/nav2.yaml), with narrower safety margins, finer costmap cells and a local costmap built from live lidar only. Movement is core to the pet, so replacing Nav2 with our own movement layer stays an open option: a safety layer every speed command goes through, reactive moves (wander, escape, wall-follow), then our own path planner.

**The "brain" runs off the robot.** The robot's Raspberry Pi 4 is busy running the robot's own software. ML runs on the Mac (PyTorch on the Apple Silicon GPU), ROS 2 runs in the VM (see [vm/README.md](vm/README.md)), and both talk to the robot over Wi-Fi. Anything streamed should use compressed image topics.

**Two kinds of novelty, kept separate.** "I haven't been / seen this here" (coverage novelty) and "I didn't expect that" (surprise) are different signals. Keeping them apart makes it possible to balance them later.

**Every change to the robot or VM is a reversible script.** The robot is expensive and easy to break, so changes go through [ops/](ops/README.md): units with status, diff, apply and revert, backups and automatic rollback, plus a health check. Network, firmware and the base's own config stay manual.

## Stages

### 1. Map the house and keep the map current

- [x] Build a map with SLAM, following the [Generating a map](https://turtlebot.github.io/turtlebot4-user-manual/tutorials/generate_map.html) tutorial.
- [x] Save the map at the end of a run (`ops/field-test.sh stop` saves it to `~/maps/` on the VM, both as an image for Nav2 and as slam_toolbox's full state).
- [ ] Navigate on a saved map, using `turtlebot4_navigation`'s `localization.launch.py` and `nav2.launch.py`.
- [ ] Keep the map up to date as furniture moves, e.g. by reloading a saved slam_toolbox map and continuing to map onto it.
- [ ] **Recover from being picked up.** SLAM tracks the robot mostly by wheel motion, so carrying it somewhere else corrupts the map. The plan:
  - [x] On pickup (`kidnap_status`, now relayed from the base), stop: cancel the goal, pause SLAM (`pause_new_measurements`) and save its state (`serialize_map`) and the grid. Built; not yet run on the robot.
  - [x] When set down, reset SLAM (`reset`; it doesn't resume a paused SLAM by itself, so the explorer resumes it), turn a full circle, and start a fresh map. Built; not yet run on the robot.
  - Repeatedly match the fresh map against the saved one (every rotation and offset, with FFT cross-correlation), keeping every placement that fits well as a hypothesis and re-scoring only those as the fresh map grows.
  - Accept a match only when one hypothesis clearly wins (e.g. above 95%) and fits well in absolute terms; then reload the saved map at that pose (`deserialize_map`). If nothing fits, the robot is somewhere new: keep the fresh map. A false match corrupts the saved map, so when in doubt, don't match.
  - While several hypotheses remain, drive to where they predict the most different views (active localization), so one look settles it.

### 2. Explore on its own

- [x] Frontier exploration: drive to the edge between mapped and unmapped space. Our own node, [src/pet_explorer](src/pet_explorer/README.md), so core behavior stays fully under our control; [m-explore-ros2](https://github.com/robo-friends/m-explore-ros2) (explore_lite, not released for Jazzy) is the reference. First full run (2026-10-30, about 4½ minutes): mapped 90% or more of the kitchen and living room.
- [x] Give up quickly on stalled goals (no progress in 8 s, Nav2 recovery, repeated bumps) and try short goals nearby until the robot moves again.
- [ ] Fewer stalls: most goals in the first run ended with "no progress in 8 s".
- [ ] Get out of spots where Nav2 thinks the robot is already touching something and won't plan at all: an escape move driven directly from the lidar scan (turn toward open space, creep out).
- [ ] Measure progress by how much of the map gets filled in.
- [ ] While exploring, record camera, position and map data to rosbags, so stages 4 and 5 have data to work with.

### 3. A simple pet routine

- [ ] A small state machine or behavior tree: explore → wander → return to the dock when the battery is low → rest → repeat.
- [ ] This is the framework later stages plug into. The robot's power saver turns the camera off while docked, so resting means not seeing.

### 4. Look at things and remember them

- [ ] Turn camera frames into feature vectors with a pretrained vision model (e.g. DINOv2 or CLIP) on the Mac.
- [ ] Store them by location on the map.
- [ ] Coverage novelty: a view is new if it's unlike anything seen before at that spot. This also catches changes ("something moved in the kitchen").

### 5. A world model

- [ ] Train a model to predict the feature vector of the next view from the current view and the planned move. Predicting features rather than raw pixels is much easier to train, and it uses the data from stage 4.
- [ ] Its prediction error is the measure of surprise.

### 6. Curiosity as a reward

- [ ] Reward choosing goals where the world model is surprised.
- [ ] Watch for the "noisy TV" problem: raw prediction error gets hooked on things that are just random (a TV, a window, a moving fan). The usual fix is to reward learning progress (surprise that shrinks as the model learns) rather than surprise itself. That also gives the pet-like behavior: drawn to new things, bored once it understands them.
- [ ] Getting its bearings after being picked up (stage 1) is the same idea in a simple form: go where you're most uncertain about what you'll see.

## Known issues

- **Camera doesn't restart after docking.** The robot's power saver stops the camera on the dock; on undocking, the start request sometimes never gets answered. `sudo systemctl restart turtlebot4` on the Pi fixes it. Turning the power saver off (as an ops unit) would avoid it.
- **Low obstacles are invisible to the lidar.** It scans one plane, so things lower than that (e.g. a metal bar on the floor) only show up as bumps. The camera's depth data could fill this in.
- **Battery.** Off the dock with the lidar and camera on, the battery went from 58% to 28% in about 50 minutes (2026-10-31); field tests refuse to start below 25%.

## Background reading

- Pathak et al., 2017, *Curiosity-driven Exploration by Self-supervised Prediction* (ICM).
- Burda et al., 2018, *Exploration by Random Network Distillation* (RND).
- Schmidhuber's work on curiosity and learning progress.
- Burgard, Fox and Thrun's work on active Markov localization (late 1990s): choosing moves that best tell apart where the robot might be.

## Not yet

- Audio: the TurtleBot 4 has no microphone. Adding one would mean a USB microphone on the Pi.
