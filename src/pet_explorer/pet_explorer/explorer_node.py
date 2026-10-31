"""Frontier exploration: repeatedly send Nav2 to the best frontier on the SLAM map until none are left.

Needs slam_toolbox publishing /map, TF from map to the robot, and Nav2's navigate_to_pose action,
e.g. turtlebot4_navigation's slam.launch.py and nav2.launch.py.

When a goal fails without the robot having moved, it tries short "probe" goals nearby, most open
space first, until one gets the robot moving, then goes back to frontiers.
"""
import math

from action_msgs.msg import GoalStatus
from irobot_create_msgs.msg import HazardDetection, HazardDetectionVector
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from pet_explorer.frontiers import (
    cell_to_world, cost, find_frontiers, frontier_mask, frontier_near, goal_cell, world_to_cell,
)
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
        # A goal that failed is avoided, along with anything this close to it (metres), for this long
        # (seconds). Other parts of the same frontier stay available.
        self.blacklist_radius = self.declare_parameter('blacklist_radius', 0.5).value
        self.blacklist_duration = self.declare_parameter('blacklist_duration', 300.0).value
        # Probe goals: points this far from the robot (metres), at this many angles, at least
        # probe_clearance (metres) from obstacles on the map. Each gets probe_timeout (seconds).
        self.probe_radii = self.declare_parameter('probe_radii', [0.4, 0.7]).value
        self.probe_angles = self.declare_parameter('probe_angles', 12).value
        self.probe_clearance = self.declare_parameter('probe_clearance', 0.2).value
        self.probe_timeout = self.declare_parameter('probe_timeout', 6.0).value
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
        self.finished = False

        # slam_toolbox publishes the map latched and reliable.
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, 'map', self.on_map, map_qos)
        self.create_subscription(HazardDetectionVector, 'hazard_detection', self.on_hazards, qos_profile_sensor_data)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
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
        if self.map is None:
            self.get_logger().info('Waiting for the map', throttle_duration_sec=5.0)
            return
        robot = self.robot_xy()
        if robot is None:
            return
        if self.goal_xy is not None:
            self.check_goal(robot)
            return
        if not self.nav.server_is_ready():
            self.get_logger().info('Waiting for Nav2 (navigate_to_pose)', throttle_duration_sec=5.0)
            return
        if self.stuck_at is not None and self.try_probe(robot):
            return
        self.go_to_frontier(robot)

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
            self.fail_goal(self.goal_start_xy, 'Nav2 rejected it', cancel=False)
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self.on_result(handle, f))

    def on_result(self, handle, future):
        if handle is not self.goal_handle:
            return  # a goal we already gave up on
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Reached {self.goal_kind}')
            self.clear_goal()
        else:
            robot = self.robot_xy() or self.goal_start_xy
            self.fail_goal(robot, f'Nav2 failed (status {status})', cancel=False)

    def fail_goal(self, robot, reason, cancel=True):
        """Give up on the current goal. A failed frontier is avoided for a while, and if the robot
        never got going, probing starts."""
        self.get_logger().info(f'Abandoning {self.goal_kind}: {reason}')
        if self.goal_kind == FRONTIER:
            until = self.get_clock().now() + Duration(seconds=self.blacklist_duration)
            self.blacklist.append((self.goal_xy, until))
            self.get_logger().info(f'Avoiding ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}) for {self.blacklist_duration:.0f} s')
            if math.dist(robot, self.goal_start_xy) <= self.progress_distance:
                self.get_logger().info('Stuck; trying nearby spots')
                self.stuck_at = robot
                self.probes_tried = []
        if cancel and self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.clear_goal()

    def clear_goal(self):
        self.goal_kind = None
        self.goal_xy = None
        self.goal_handle = None
        self.goal_sent_at = None

    def stop(self):
        """Cancel the current goal so the robot doesn't keep driving after the explorer exits."""
        if self.goal_handle is not None:
            self.get_logger().info('Cancelling the current goal')
            rclpy.spin_until_future_complete(self, self.goal_handle.cancel_goal_async(), timeout_sec=2.0)


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
