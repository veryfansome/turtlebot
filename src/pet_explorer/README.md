# pet_explorer

Frontier exploration for the TurtleBot 4 (stage 2 of the [roadmap](../../ROADMAP.md)). The explorer watches the SLAM map, finds frontiers (edges between mapped free space and unknown space), and sends Nav2 to the best one, over and over, until nothing unexplored is reachable.

Written from scratch rather than depending on [m-explore-ros2](https://github.com/robo-friends/m-explore-ros2), so the behavior stays fully under our control. The approach and default weights follow its `explore_lite`.

## How it works

1. Every second (`tick_period`) it takes the latest `/map` and the robot's position from TF (`map` → `base_link`).
2. Frontier cells are free cells with an unknown cell directly next to them. Touching ones are grouped into frontiers, and frontiers shorter than `min_frontier_size` are ignored.
3. Each frontier gets a cost of `distance_weight × distance − size_weight × length`, so near, long frontiers win. This is `cost()` in `pet_explorer/frontiers.py`, the function later stages replace with novelty scores.
4. The cheapest one goes to Nav2's `navigate_to_pose`.
5. A goal is dropped if the frontier gets mapped before the robot arrives (it moves on, no penalty). It's also dropped, and the area within `blacklist_radius` of that spot avoided for `blacklist_duration`, if Nav2 fails or rejects it, it takes longer than `goal_timeout`, the robot moves less than `progress_distance` in `progress_timeout` (stuck or spinning), or it bumps into something `max_bumps` times on the way (from the base's `/hazard_detection`). Low obstacles sit under the lidar's scan plane, so Nav2 can't see them and would keep driving into them.
6. With no frontiers left it logs `exploration complete` and waits, picking up again if the map grows.

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

In separate VM terminals:

```bash
ros2 launch turtlebot4_navigation slam.launch.py
ros2 launch turtlebot4_navigation nav2.launch.py
ros2 launch turtlebot4_viz view_navigation.launch.py   # optional: watch the map and goals in RViz
ros2 run pet_explorer explorer
```

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
| `progress_timeout` | `20.0` | Give up on a goal after this long without progress (s) |
| `max_bumps` | `2` | Give up on a goal after this many bumps |
| `blacklist_radius` | `0.5` | Avoid goals this close to a failed one; the rest of that frontier stays available. Also how near a frontier must stay to keep the current goal (m) |
| `blacklist_duration` | `300.0` | How long a failed spot is avoided (s) |
| `tick_period` | `1.0` | How often to re-check the map and goal (s) |
