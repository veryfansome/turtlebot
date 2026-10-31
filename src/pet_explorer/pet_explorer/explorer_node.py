"""Frontier exploration: repeatedly send Nav2 to the best frontier on the SLAM map until none are left.

Needs slam_toolbox publishing /map, TF from map to the robot, and Nav2's navigate_to_pose action,
e.g. turtlebot4_navigation's slam.launch.py and nav2.launch.py.

When a goal fails without the robot having moved, it backs up a little, then tries short "probe"
goals nearby, most open space first, until one gets the robot moving, then goes back to frontiers.
If the lidar goes quiet it cancels its goal and holds still until scans come back.

When the robot is picked up (the base's kidnap_status), it stops, pauses SLAM and saves the map
(SLAM's state and the grid) to map_dir. When it's set down, it resets SLAM to an empty map, turns
a full circle and explores the fresh map: SLAM tracks the robot mostly by wheel motion, so after
being carried somewhere its old map no longer matches where it is.
"""
from datetime import datetime
import math
import os

from action_msgs.msg import GoalStatus
from irobot_create_msgs.msg import HazardDetection, HazardDetectionVector, KidnapStatus
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import BackUp, NavigateToPose, Spin
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rcl_interfaces.srv import GetParameters
from rclpy.time import Time
from slam_toolbox.srv import Pause, Reset, SerializePoseGraph
from tf2_ros import Buffer, TransformException, TransformListener

from pet_explorer.frontiers import (
    cell_to_world, cost, find_frontiers, frontier_mask, frontier_near, goal_cell, world_to_cell,
)
from pet_explorer.maps import save_grid
from pet_explorer.probes import probe_goals

FRONTIER, PROBE = 'frontier', 'probe'


class Explorer(Node):

    def __init__(self):
        super().__init__('explorer')
        self.global_frame = self.declare_parameter('global_frame', 'map').value
        self.robot_frame = self.declare_parameter('robot_base_frame', 'base_link').value
        self.min_frontier_size = self.declare_parameter('min_frontier_size', 0.5).value  # metres
        self.distance_weight = self.declare_parameter('distance_weight', 3.0).value
        self.size_weight = self.declare_parameter('size_weight', 1.0).value
        self.goal_timeout = self.declare_parameter('goal_timeout', 60.0).value  # seconds
        # Give up if the robot hasn't moved progress_distance (metres) in progress_timeout (seconds),
        # e.g. stuck or spinning in place.
        self.progress_distance = self.declare_parameter('progress_distance', 0.25).value
        self.progress_timeout = self.declare_parameter('progress_timeout', 8.0).value
        # Give up after this many bumps on the way to one goal. Low obstacles sit under the lidar's
        # scan plane, so Nav2 can't see them and keeps driving into them.
        self.max_bumps = self.declare_parameter('max_bumps', 2).value
        # A goal that failed or was reached is avoided, along with anything this close to it (metres),
        # for this long (seconds). Other parts of the same frontier stay available. Reached ones too,
        # because a frontier whose far side can't be seen from there stays a frontier.
        self.blacklist_radius = self.declare_parameter('blacklist_radius', 0.5).value
        self.blacklist_duration = self.declare_parameter('blacklist_duration', 300.0).value
        # Probe goals: points this far from the robot (metres), at this many angles, at least
        # probe_clearance (metres) from obstacles on the map. Each gets probe_timeout (seconds).
        self.probe_radii = self.declare_parameter('probe_radii', [0.4, 0.7]).value
        self.probe_angles = self.declare_parameter('probe_angles', 12).value
        self.probe_clearance = self.declare_parameter('probe_clearance', 0.2).value
        self.probe_timeout = self.declare_parameter('probe_timeout', 6.0).value
        # Before probing, back straight up this far (metres; 0 to skip): the way it came was clear,
        # and it gets the robot off things the lidar can't see (a cushion, a floor strip).
        self.backup_distance = self.declare_parameter('backup_distance', 0.15).value
        self.backup_speed = self.declare_parameter('backup_speed', 0.1).value  # m/s
        # Hold still if no lidar scan has arrived for this long (seconds): Nav2 is blind then.
        self.scan_timeout = self.declare_parameter('scan_timeout', 3.0).value
        # Pickup handling: where maps are saved, how long to wait after being set down before
        # starting a fresh map (seconds), and whether to turn a full circle first.
        self.map_dir = os.path.expanduser(self.declare_parameter('map_dir', '~/maps').value)
        self.settle_time = self.declare_parameter('settle_time', 3.0).value
        self.spin_on_landing = self.declare_parameter('spin_on_landing', True).value
        tick_period = self.declare_parameter('tick_period', 0.5).value  # seconds

        self.map = None  # (grid, origin_xy, resolution)
        self.blacklist = []  # (xy, time it expires)
        self.goal_kind = None  # FRONTIER or PROBE
        self.goal_xy = None  # set from sending a goal until it finishes or is cancelled
        self.goal_handle = None  # set once Nav2 accepts the goal
        self.goal_sent_at = None
        self.goal_start_xy = None  # where the robot was when the goal was sent
        self.progress_xy = None  # where the robot was when it last made progress, and when
        self.progress_at = None
        self.recovering = False  # Nav2 has started recovery behaviours for the current goal
        self.bumps = 0
        self.bumping = False
        self.stuck_at = None  # set while probing: where the robot got stuck
        self.probes_tried = []
        self.backup_handle = None  # set while backing up, before probing
        self.backing_up = False
        self.last_scan_at = None
        self.scan_silent = False
        self.finished = False
        self.nav_active = False
        self.nav_state_future = None
        self.robot_last = None  # the last known robot position, saved with the map on pickup
        self.lifted = False
        self.landing = None  # after being set down: 'settling', 'resetting', 'mapping', 'spinning'
        self.landed_at = None
        self.spin_handle = None
        self.pickups = 0
        self.session_dir = os.path.join(self.map_dir, 'explorer-' + datetime.now().strftime('%Y%m%d-%H%M%S'))

        # slam_toolbox publishes the map latched and reliable.
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, 'map', self.on_map, map_qos)
        self.create_subscription(HazardDetectionVector, 'hazard_detection', self.on_hazards, qos_profile_sensor_data)
        self.create_subscription(KidnapStatus, 'kidnap_status', self.on_kidnap, qos_profile_sensor_data)
        self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        # Nav2's action server appears before Nav2 is active, and rejects goals until then.
        self.nav_state = self.create_client(GetState, 'bt_navigator/get_state')
        self.spin = ActionClient(self, Spin, 'spin')
        self.backup = ActionClient(self, BackUp, 'backup')
        self.slam_params = self.create_client(GetParameters, 'slam_toolbox/get_parameters')
        self.slam_pause = self.create_client(Pause, 'slam_toolbox/pause_new_measurements')
        self.slam_reset = self.create_client(Reset, 'slam_toolbox/reset')
        self.slam_serialize = self.create_client(SerializePoseGraph, 'slam_toolbox/serialize_map')
        # Nav2's costmaps keep obstacles until a lidar beam passes through them, so marks from mid-air
        # or from someone who stood nearby can box the robot in for good.
        self.costmap_clearers = [
            self.create_client(ClearEntireCostmap, 'local_costmap/clear_entirely_local_costmap'),
            self.create_client(ClearEntireCostmap, 'global_costmap/clear_entirely_global_costmap'),
        ]
        self.create_timer(tick_period, self.tick)

    def on_map(self, msg: OccupancyGrid):
        grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map = (grid, origin, msg.info.resolution)

    def on_hazards(self, msg: HazardDetectionVector):
        bumping = any(d.type == HazardDetection.BUMP for d in msg.detections)
        if bumping and not self.bumping and self.goal_handle is not None:
            self.bumps += 1
            self.get_logger().warn(f'Bumped into something ({self.bumps}/{self.max_bumps})')
        self.bumping = bumping

    def on_kidnap(self, msg: KidnapStatus):
        if msg.is_kidnapped and not self.lifted:
            self.lifted = True
            self.on_lifted()
        elif not msg.is_kidnapped and self.lifted:
            self.lifted = False
            self.landed_at = self.get_clock().now()
            self.landing = 'settling'
            self.get_logger().warn(f'Set down; starting a fresh map in {self.settle_time:.0f} s')

    def on_scan(self, msg: LaserScan):
        self.last_scan_at = self.get_clock().now()
        if self.scan_silent:
            self.scan_silent = False
            self.get_logger().info('Lidar scans are back')

    def scan_ok(self):
        """Whether lidar scans are arriving. Cancels everything when they stop: Nav2 is blind then."""
        if self.last_scan_at is None:
            self.get_logger().info('Waiting for the lidar', throttle_duration_sec=5.0)
            return False
        silent = (self.get_clock().now() - self.last_scan_at).nanoseconds / 1e9
        if silent <= self.scan_timeout:
            return True
        if not self.scan_silent:
            self.scan_silent = True
            self.get_logger().warn(f'Lidar silent for {silent:.0f} s: stopping and holding still until scans come back')
            self.halt()
        else:
            self.get_logger().warn(f'Lidar still silent ({silent:.0f} s)', throttle_duration_sec=10.0)
        return False

    def halt(self):
        """Cancel whatever the robot is doing and forget being stuck."""
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        if self.backup_handle is not None:
            self.backup_handle.cancel_goal_async()
            self.backup_handle = None
        self.backing_up = False
        self.clear_goal()
        self.stuck_at = None
        self.probes_tried = []

    def on_lifted(self):
        """Stop, pause SLAM so the view from mid-air doesn't go into the map, and save the map."""
        self.pickups += 1
        self.get_logger().warn(f'Picked up (#{self.pickups}): stopping, pausing SLAM and saving the map')
        self.halt()
        if self.spin_handle is not None:
            self.spin_handle.cancel_goal_async()
            self.spin_handle = None
        self.landing = None
        base = self.session_path(f'pickup-{self.pickups}-before')
        self.save_map_grid(base + '.npz')
        self.set_slam_paused(True)
        if self.slam_serialize.service_is_ready():
            self.slam_serialize.call_async(SerializePoseGraph.Request(filename=base)).add_done_callback(
                lambda f: self.log_result(f, f'Saved SLAM state to {base}.posegraph', 'SLAM serialize_map failed'))
        else:
            self.get_logger().error('slam_toolbox/serialize_map not available; SLAM state not saved')

    def set_slam_paused(self, paused):
        """Pause or resume SLAM taking in scans. Its pause service toggles, so this checks the current
        state first. (Its reset service doesn't resume a paused SLAM, whatever it's asked.)"""
        word = 'paused' if paused else 'resumed'
        if not self.slam_params.service_is_ready():
            self.get_logger().error(f'slam_toolbox not available; SLAM not {word}')
            return

        def on_params(future):
            values = future.result().values if future.result() else []
            if not values or values[0].type != 1:  # 1: bool
                self.get_logger().error("Can't read SLAM's paused_new_measurements; not toggling it")
            elif values[0].bool_value == paused:
                self.get_logger().info(f'SLAM already {word}')
            else:
                self.slam_pause.call_async(Pause.Request()).add_done_callback(
                    lambda f: self.log_result(f, f'SLAM {word}', f'SLAM toggle to {word} failed', ok=lambda r: r.status))

        request = GetParameters.Request(names=['paused_new_measurements'])
        self.slam_params.call_async(request).add_done_callback(on_params)

    def continue_landing(self):
        """After being set down: settle, reset SLAM, wait for the fresh map, turn a full circle."""
        if self.landing == 'settling':
            if (self.get_clock().now() - self.landed_at).nanoseconds / 1e9 < self.settle_time:
                return
            if not self.slam_reset.service_is_ready():
                self.get_logger().error('slam_toolbox/reset not available; carrying on with the old map')
                self.landing = None
                return
            self.landing = 'resetting'
            self.slam_reset.call_async(Reset.Request(pause_new_measurements=False)).add_done_callback(self.on_reset)
        elif self.landing == 'mapping' and self.map is not None and self.robot_xy() is not None:
            if not self.spin_on_landing or not self.spin.server_is_ready():
                self.landing = None
                self.get_logger().info('Exploring the fresh map')
                return
            self.landing = 'spinning'
            self.get_logger().info('Turning a full circle to start the fresh map')
            goal = Spin.Goal(target_yaw=2 * math.pi)
            goal.time_allowance.sec = 30
            self.spin.send_goal_async(goal).add_done_callback(self.on_spin_accepted)

    def on_reset(self, future):
        result = future.result()
        if result is None or result.result != Reset.Response.RESULT_SUCCESS:
            self.get_logger().error('SLAM reset failed; carrying on with the old map')
            self.landing = None
            return
        # The old map, and positions on it, mean nothing now.
        self.map = None
        self.blacklist = []
        self.landing = 'mapping'
        self.get_logger().info('SLAM reset: starting a fresh map')
        self.set_slam_paused(False)
        self.clear_costmaps('of what the lidar saw in mid-air')

    def clear_costmaps(self, why):
        """Wipe Nav2's local and global costmaps; they refill from the next lidar scans and the map."""
        self.get_logger().info(f'Clearing the costmaps {why}')
        for client in self.costmap_clearers:
            if client.service_is_ready():
                client.call_async(ClearEntireCostmap.Request()).add_done_callback(
                    lambda f, name=client.srv_name: self.log_result(f, f'Cleared {name}', f'{name} failed', ok=lambda r: True))
            else:
                self.get_logger().error(f'{client.srv_name} not available; costmap not cleared')

    def on_spin_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Spin rejected; exploring the fresh map')
            self.landing = None
            return
        self.spin_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self.on_spin_done(handle, f))

    def on_spin_done(self, handle, future):
        if handle is not self.spin_handle:
            return  # cancelled by another pickup
        self.spin_handle = None
        self.landing = None
        status = future.result().status
        message = 'Turned a full circle' if status == GoalStatus.STATUS_SUCCEEDED else f'Spin ended early (status {status})'
        self.get_logger().info(f'{message}; exploring the fresh map')

    def session_path(self, name):
        os.makedirs(self.session_dir, exist_ok=True)
        return os.path.join(self.session_dir, name)

    def save_map_grid(self, path):
        if self.map is None:
            self.get_logger().warn('No map to save yet')
            return
        grid, origin, resolution = self.map
        robot = self.robot_last if self.robot_last is not None else (math.nan, math.nan)
        save_grid(path, grid, origin, resolution, robot_xy=robot)
        self.get_logger().info(f'Saved map grid to {path}')

    def log_result(self, future, success, failure, ok=lambda r: r.result == 0):
        result = future.result()
        if result is not None and ok(result):
            self.get_logger().info(success)
        else:
            self.get_logger().error(failure)

    def on_feedback(self, msg):
        # Nav2 only starts recovery behaviours (clearing costmaps, spinning, backing up) once it
        # can't plan or follow a path, which on this robot has meant it's stuck.
        if (self.goal_handle is not None and bytes(msg.goal_id.uuid) == bytes(self.goal_handle.goal_id.uuid)
                and msg.feedback.number_of_recoveries > 0):
            self.recovering = True

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame, self.robot_frame, Time())
        except TransformException as e:
            self.get_logger().warn(f'No robot pose yet: {e}', throttle_duration_sec=5.0)
            return None
        return (t.transform.translation.x, t.transform.translation.y)

    def tick(self):
        if self.lifted:
            self.get_logger().info('Picked up; waiting to be set down', throttle_duration_sec=5.0)
            return
        if self.landing is not None:
            self.continue_landing()
            return
        if not self.scan_ok():
            return
        if self.map is None:
            self.get_logger().info('Waiting for the map', throttle_duration_sec=5.0)
            return
        robot = self.robot_xy()
        if robot is None:
            return
        self.robot_last = robot
        if self.backing_up:
            return
        if self.goal_xy is not None:
            self.check_goal(robot)
            return
        if not self.nav2_active():
            self.get_logger().info('Waiting for Nav2 to become active', throttle_duration_sec=5.0)
            return
        if self.stuck_at is not None and self.try_probe(robot):
            return
        self.go_to_frontier(robot)

    def nav2_active(self):
        """Whether Nav2's bt_navigator is active, checked without blocking until it first is."""
        if self.nav_active:
            return self.nav.server_is_ready()
        if self.nav_state_future is not None and self.nav_state_future.done():
            result = self.nav_state_future.result()
            self.nav_active = result is not None and result.current_state.id == State.PRIMARY_STATE_ACTIVE
            self.nav_state_future = None
            if self.nav_active:
                self.get_logger().info('Nav2 is active')
                return self.nav.server_is_ready()
        if self.nav_state_future is None and self.nav_state.service_is_ready():
            self.nav_state_future = self.nav_state.call_async(GetState.Request())
        return False

    def avoid(self, xy):
        until = self.get_clock().now() + Duration(seconds=self.blacklist_duration)
        self.blacklist.append((xy, until))
        self.get_logger().info(f'Avoiding ({xy[0]:.2f}, {xy[1]:.2f}) for {self.blacklist_duration:.0f} s')

    def try_probe(self, robot):
        """While stuck, send the next probe goal. Returns False once unstuck or out of probes."""
        if math.dist(robot, self.stuck_at) > self.progress_distance:
            self.get_logger().info('Moving again; back to frontiers')
        else:
            grid, origin, resolution = self.map
            for xy in probe_goals(grid, origin, resolution, robot, self.probe_radii, self.probe_angles, self.probe_clearance):
                if all(math.dist(xy, tried) > 0.15 for tried in self.probes_tried):
                    self.probes_tried.append(xy)
                    self.send_goal(PROBE, xy, robot, f'Trying nearby spot ({xy[0]:.2f}, {xy[1]:.2f})')
                    return True
            self.get_logger().warn(f'Tried {len(self.probes_tried)} nearby spots without moving; back to frontiers')
        self.stuck_at = None
        self.probes_tried = []
        return False

    def go_to_frontier(self, robot):
        now = self.get_clock().now()
        self.blacklist = [(xy, until) for xy, until in self.blacklist if until > now]
        avoid = [xy for xy, _ in self.blacklist]
        grid, origin, resolution = self.map
        min_cells = max(1, round(self.min_frontier_size / resolution))
        candidates = []
        for frontier in find_frontiers(grid, min_cells):
            cell = goal_cell(frontier, avoid, self.blacklist_radius, origin, resolution)
            if cell is None:
                continue
            xy = cell_to_world(cell, origin, resolution)
            size_m = frontier.size * resolution
            candidates.append((cost(xy, size_m, robot, self.distance_weight, self.size_weight), xy, size_m))
        if not candidates:
            if not self.finished:
                self.get_logger().info('No frontiers left: exploration complete')
                self.finished = True
            return
        self.finished = False
        _, xy, size_m = min(candidates)
        self.send_goal(FRONTIER, xy, robot, f'Heading to frontier at ({xy[0]:.2f}, {xy[1]:.2f}), {size_m:.2f} m long')

    def check_goal(self, robot):
        """Give up on the current goal if it's stuck, taking too long, or its frontier has been mapped."""
        if self.goal_handle is None:
            return  # Nav2 hasn't accepted it yet
        now = self.get_clock().now()
        timeout = self.probe_timeout if self.goal_kind == PROBE else self.goal_timeout
        if (now - self.goal_sent_at).nanoseconds / 1e9 > timeout:
            self.fail_goal(robot, 'timed out')
            return
        if self.recovering:
            self.fail_goal(robot, 'Nav2 started recovery')
            return
        if self.bumps >= self.max_bumps:
            self.fail_goal(robot, f'bumped into something {self.bumps} times')
            return
        if math.dist(robot, self.progress_xy) > self.progress_distance:
            self.progress_xy, self.progress_at = robot, now
        elif (now - self.progress_at).nanoseconds / 1e9 > self.progress_timeout:
            self.fail_goal(robot, f'no progress in {self.progress_timeout:.0f} s')
            return
        if self.goal_kind == FRONTIER:
            grid, origin, resolution = self.map
            cell = world_to_cell(self.goal_xy, origin, resolution)
            radius = max(1, round(self.blacklist_radius / resolution))
            if not frontier_near(frontier_mask(grid), cell, radius):
                self.get_logger().info('Abandoning frontier: already mapped')
                self.goal_handle.cancel_goal_async()
                self.clear_goal()

    def send_goal(self, kind, xy, robot, message):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.global_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = xy
        # Face the direction of travel so Nav2 doesn't spin on arrival.
        yaw = math.atan2(xy[1] - robot[1], xy[0] - robot[0])
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        self.get_logger().info(message)
        self.goal_kind = kind
        self.goal_xy = xy
        self.goal_sent_at = self.progress_at = self.get_clock().now()
        self.goal_start_xy = self.progress_xy = robot
        self.recovering = False
        self.bumps = 0
        self.nav.send_goal_async(goal, feedback_callback=self.on_feedback).add_done_callback(self.on_goal_response)

    def on_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            # Not a sign of being stuck: Nav2 never tried.
            self.fail_goal(self.goal_start_xy, 'Nav2 rejected it', cancel=False, maybe_stuck=False)
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self.on_result(handle, f))

    def on_result(self, handle, future):
        if handle is not self.goal_handle:
            return  # a goal we already gave up on
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Reached {self.goal_kind}')
            if self.goal_kind == FRONTIER:
                self.avoid(self.goal_xy)
            self.clear_goal()
        else:
            robot = self.robot_xy() or self.goal_start_xy
            self.fail_goal(robot, f'Nav2 failed (status {status})', cancel=False)

    def fail_goal(self, robot, reason, cancel=True, maybe_stuck=True):
        """Give up on the current goal. A failed frontier is avoided for a while, and if the robot
        never got going, probing starts."""
        self.get_logger().info(f'Abandoning {self.goal_kind}: {reason}')
        if self.goal_kind == FRONTIER:
            self.avoid(self.goal_xy)
            if maybe_stuck and math.dist(robot, self.goal_start_xy) <= self.progress_distance:
                self.get_logger().info('Stuck; trying nearby spots')
                self.stuck_at = robot
                self.probes_tried = []
                self.clear_costmaps('in case old obstacles are boxing it in')
                self.start_backup()
        if cancel and self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.clear_goal()

    def start_backup(self):
        """Back straight up a little; probing starts when it's done (or couldn't be done)."""
        if self.backup_distance <= 0:
            return
        if not self.backup.server_is_ready():
            self.get_logger().warn("Nav2's backup isn't available; probing without backing up")
            return
        goal = BackUp.Goal()
        goal.target.x = float(self.backup_distance)
        goal.speed = float(self.backup_speed)
        goal.time_allowance.sec = math.ceil(3 * self.backup_distance / self.backup_speed)
        self.backing_up = True
        self.get_logger().info(f'Backing up {self.backup_distance:.2f} m')
        self.backup.send_goal_async(goal).add_done_callback(self.on_backup_accepted)

    def on_backup_accepted(self, future):
        handle = future.result()
        if not self.backing_up:  # halted in the meantime
            if handle.accepted:
                handle.cancel_goal_async()
            return
        if not handle.accepted:
            self.get_logger().warn('Backup rejected; probing')
            self.backing_up = False
            return
        self.backup_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self.on_backup_done(handle, f))

    def on_backup_done(self, handle, future):
        if handle is not self.backup_handle:
            return  # cancelled
        self.backup_handle = None
        self.backing_up = False
        result = future.result()
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Backed up; trying nearby spots')
        else:
            reason = result.result.error_msg or f'status {result.status}'
            self.get_logger().info(f"Couldn't back up ({reason}); trying nearby spots")

    def clear_goal(self):
        self.goal_kind = None
        self.goal_xy = None
        self.goal_handle = None
        self.goal_sent_at = None

    def stop(self):
        """Cancel the current goal so the robot doesn't keep driving after the explorer exits, and
        save the map grid."""
        for handle in (self.goal_handle, self.spin_handle):
            if handle is not None:
                self.get_logger().info('Cancelling the current goal')
                rclpy.spin_until_future_complete(self, handle.cancel_goal_async(), timeout_sec=2.0)
        self.save_map_grid(self.session_path('final.npz'))


def main():
    # Handle Ctrl-C ourselves so the context is still up to cancel the goal.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Explorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
