# Roadmap: a TurtleBot 4 that explores the house on its own

The goal is a robotic pet: a TurtleBot 4 that moves around the house on its own initiative, notices things, and gets more interesting as it learns. This is a fun ML project, built up in stages, with a working robot at the end of each one.

## Design decisions

**Nav2 handles movement; the learning decides where to go.** Nav2 already plans paths, avoids obstacles and docks. Learning low-level driving on a real robot needs far too many trials and is hard on the furniture. Instead, the learned part picks goals ("go look over there", "go back to that spot", "rest on the dock") and Nav2 carries them out. That keeps experiments safe and quick, and it's where the pet-like behavior lives anyway.

**The "brain" runs off the robot.** The robot's Raspberry Pi 4 is busy running the robot's own software. ML runs on the Mac (PyTorch on the Apple Silicon GPU), ROS 2 runs in the VM (see [vm/README.md](vm/README.md)), and both talk to the robot over Wi-Fi. Anything streamed should use compressed image topics.

**Two kinds of novelty, kept separate.** "I haven't been / seen this here" (coverage novelty) and "I didn't expect that" (surprise) are different signals. Keeping them apart makes it possible to balance them later.

## Stages

### 1. Map the house and keep the map current

- [x] Build a map with SLAM, following the [Generating a map](https://turtlebot.github.io/turtlebot4-user-manual/tutorials/generate_map.html) tutorial.
- [ ] Save the map, then navigate with it using `turtlebot4_navigation`'s `localization.launch.py` and `nav2.launch.py`.
- [ ] Keep the map up to date as furniture moves, e.g. by reloading a saved slam_toolbox map and continuing to map onto it.

### 2. Explore on its own

- [ ] Frontier exploration: drive to the edge between mapped and unmapped space. [m-explore-ros2](https://github.com/robo-friends/m-explore-ros2) (explore_lite) does this on top of Nav2; check whether it needs building from source for Jazzy.
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

## Background reading

- Pathak et al., 2017, *Curiosity-driven Exploration by Self-supervised Prediction* (ICM).
- Burda et al., 2018, *Exploration by Random Network Distillation* (RND).
- Schmidhuber's work on curiosity and learning progress.

## Not yet

- Audio: the TurtleBot 4 has no microphone. Adding one would mean a USB microphone on the Pi.
