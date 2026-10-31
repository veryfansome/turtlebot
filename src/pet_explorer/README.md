# pet_explorer

Frontier exploration for the TurtleBot 4 (stage 2 of the [roadmap](../../ROADMAP.md)). The explorer watches the SLAM map, finds frontiers (edges between mapped free space and unknown space), and sends Nav2 to the best one, over and over, until nothing unexplored is reachable.

Written from scratch rather than depending on [m-explore-ros2](https://github.com/robo-friends/m-explore-ros2), so the behavior stays fully under our control. The approach and default weights follow its `explore_lite`.

## How it works

1. Every half second (`tick_period`) it takes the latest `/map` and the robot's position from TF (`map` → `base_link`).
2. Frontier cells are free cells with an unknown cell directly next to them. Touching ones are grouped into frontiers, and frontiers shorter than `min_frontier_size` are ignored.
3. Each frontier gets a cost of `distance_weight × distance − size_weight × length`, so near, long frontiers win. This is `cost()` in `pet_explorer/frontiers.py`, the function later stages replace with novelty scores.
4. The cheapest one goes to Nav2's `navigate_to_pose`.
5. A goal is dropped if the frontier gets mapped before the robot arrives (it moves on, no penalty). It's given up, and the area within `blacklist_radius` of that spot avoided for `blacklist_duration`, if Nav2 fails or rejects it, Nav2 starts its recovery behaviours (clearing costmaps, spinning, backing up: on this robot a sign it's stuck), it takes longer than `goal_timeout`, the robot moves less than `progress_distance` in `progress_timeout`, or it bumps into something `max_bumps` times on the way (from the base's `/hazard_detection`; low obstacles sit under the lidar's scan plane, so Nav2 can't see them).
   If the robot never got going (it's still within `progress_distance` of where the goal started), it clears Nav2's costmaps (obstacles stay marked until a lidar beam passes through them, so someone who stood near the robot can leave a lasting phantom wall) and tries short probe goals nearby in quick succession: points `probe_radii` away at `probe_angles` angles, on known free space at least `probe_clearance` from obstacles, most open first, each given `probe_timeout`. As soon as the robot has moved `progress_distance` it goes back to frontiers; if every probe fails it goes back to frontiers anyway.
6. A reached frontier is avoided for `blacklist_duration` too: when its far side can't be seen from there, it stays a frontier, and the explorer would otherwise pick it again and again.
7. **Picked up** (the base's `kidnap_status`, relayed by the `pi/create3-relays` ops unit): it cancels its goal, pauses SLAM so the view from mid-air doesn't go into the map, and saves the map to `map_dir/explorer-<start time>/pickup-<n>-before`: the grid as `.npz` (with the robot's last position) and SLAM's state as `.posegraph`/`.data`. **Set down**: after `settle_time` it resets SLAM to an empty map, resumes it, clears Nav2's costmaps (the tilted lidar marks floor and ceiling as obstacles while it's in the air), turns a full circle (`spin_on_landing`) and explores the fresh map. SLAM tracks the robot mostly by wheel motion, so after being carried its old map no longer fits; matching the fresh map against the saved one is the next step (see the roadmap). On exit it also saves `final.npz`.
8. With no frontiers left it logs `exploration complete` and waits, picking up again if the map grows.

Ctrl-C cancels the current Nav2 goal before exiting, so the robot stops.

## Building (in the VM)

The repo root is a colcon workspace, with packages under `src/`. A VM made with [vm/create-vm.sh](../../vm/README.md) already has it cloned to `~/turtlebot`, built, and sourced in new terminals. After pulling changes:

```bash
cd ~/turtlebot
colcon build --symlink-install
source install/setup.bash
```

Unit tests for the frontier logic, which need only numpy and pytest:

```bash
colcon test --packages-select pet_explorer && colcon test-result --verbose
```

## Running

Undock the robot first: the power saver keeps the lidar and camera off while it's docked. Clear the floor of cables, and block off stairs (the base has cliff sensors, but don't rely on them on the first runs). Stay nearby.

Start SLAM, Nav2 and the explorer together:

```bash
ros2 launch pet_explorer explore.launch.py rviz:=true
```

`rviz:=true` also opens RViz to watch the map and goals. `nav2_params:=...` and `slam_params:=...` swap in other config files.

Nav2 runs with this repo's [config/nav2.yaml](../../config/nav2.yaml), TurtleBot 4's settings with narrower safety margins so it fits through tighter gaps.

To stop: Ctrl-C the explorer. Save the map before closing SLAM, as in the [Generating a map](https://turtlebot.github.io/turtlebot4-user-manual/tutorials/generate_map.html) tutorial.

## Parameters

Set with `--ros-args -p name:=value`.

| Parameter | Default | Meaning |
|---|---|---|
| `global_frame` | `map` | Map frame |
| `robot_base_frame` | `base_link` | Robot frame |
| `min_frontier_size` | `0.5` | Ignore frontiers shorter than this (m) |
| `distance_weight` | `3.0` | Cost per metre of distance to the frontier |
| `size_weight` | `1.0` | Cost reduction per metre of frontier length |
| `goal_timeout` | `60.0` | Give up on a goal after this long (s) |
| `progress_distance` | `0.25` | Movement that counts as progress (m) |
| `progress_timeout` | `8.0` | Give up on a goal after this long without progress (s) |
| `max_bumps` | `2` | Give up on a goal after this many bumps |
| `blacklist_radius` | `0.5` | Avoid goals this close to a failed or reached one; the rest of that frontier stays available. Also how near a frontier must stay to keep the current goal (m) |
| `blacklist_duration` | `300.0` | How long a failed or reached spot is avoided (s) |
| `probe_radii` | `[0.4, 0.7]` | Distances of probe goals from the robot (m) |
| `probe_angles` | `12` | Probe goals per distance, evenly spaced around the robot |
| `probe_clearance` | `0.2` | Minimum distance from probe goals to obstacles on the map (m) |
| `probe_timeout` | `6.0` | Give up on a probe goal after this long (s) |
| `map_dir` | `~/maps` | Where pickup and final maps are saved |
| `settle_time` | `3.0` | Wait after being set down before starting a fresh map (s) |
| `spin_on_landing` | `true` | Turn a full circle after being set down |
| `tick_period` | `0.5` | How often to re-check the map and goal (s) |
