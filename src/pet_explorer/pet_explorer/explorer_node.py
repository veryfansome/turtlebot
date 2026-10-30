"""Frontier exploration: repeatedly send Nav2 to the best frontier on the SLAM map until none are left.

Needs slam_toolbox publishing /map, TF from map to the robot, and Nav2's navigate_to_pose action,
e.g. turtlebot4_navigation's slam.launch.py and nav2.launch.py.
"""
import math

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from pet_explorer.frontiers import cell_to_world, cost, find_frontiers, frontier_mask, frontier_near, world_to_cell


class Explorer(Node):

    def __init__(self):
        super().__init__('explorer')
        self.global_frame = self.declare_parameter('global_frame', 'map').value
        self.robot_frame = self.declare_parameter('robot_base_frame', 'base_link').value
        self.min_frontier_size = self.declare_parameter('min_frontier_size', 0.5).value  # metres
        self.distance_weight = self.declare_parameter('distance_weight', 3.0).value
        self.size_weight = self.declare_parameter('size_weight', 1.0).value
        self.goal_timeout = self.declare_parameter('goal_timeout', 60.0).value  # seconds
        # Goals that failed are skipped from then on, along with anything this close to them (metres).
        self.blacklist_radius = self.declare_parameter('blacklist_radius', 0.5).value
        tick_period = self.declare_parameter('tick_period', 1.0).value  # seconds

        self.map = None  # (grid, origin_xy, resolution)
        self.blacklist = []
        self.goal_xy = None  # set from sending a goal until it finishes or is cancelled
        self.goal_handle = None  # set once Nav2 accepts the goal
        self.goal_sent_at = None
        self.finished = False

        # slam_toolbox publishes the map latched and reliable.
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, 'map', self.on_map, map_qos)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.create_timer(tick_period, self.tick)

    def on_map(self, msg: OccupancyGrid):
        grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map = (grid, origin, msg.info.resolution)

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
            self.check_goal()
            return
        if not self.nav.server_is_ready():
            self.get_logger().info('Waiting for Nav2 (navigate_to_pose)', throttle_duration_sec=5.0)
            return

        grid, origin, resolution = self.map
        min_cells = max(1, round(self.min_frontier_size / resolution))
        candidates = []
        for frontier in find_frontiers(grid, min_cells):
            xy = cell_to_world(frontier.goal, origin, resolution)
            if any(math.dist(xy, b) < self.blacklist_radius for b in self.blacklist):
                continue
            size_m = frontier.size * resolution
            candidates.append((cost(xy, size_m, robot, self.distance_weight, self.size_weight), xy, size_m))
        if not candidates:
            if not self.finished:
                self.get_logger().info('No frontiers left: exploration complete')
                self.finished = True
            return
        self.finished = False
        _, xy, size_m = min(candidates)
        self.send_goal(xy, robot, size_m)

    def check_goal(self):
        """Give up on the current goal if it's taking too long or its frontier has been mapped."""
        if self.goal_handle is None:
            return  # Nav2 hasn't accepted it yet
        if (self.get_clock().now() - self.goal_sent_at).nanoseconds / 1e9 > self.goal_timeout:
            self.cancel_goal('timed out', blacklist=True)
            return
        grid, origin, resolution = self.map
        cell = world_to_cell(self.goal_xy, origin, resolution)
        radius = max(1, round(self.blacklist_radius / resolution))
        if not frontier_near(frontier_mask(grid), cell, radius):
            self.cancel_goal('frontier already mapped', blacklist=False)

    def send_goal(self, xy, robot, size_m):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.global_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = xy
        # Face the direction of travel so Nav2 doesn't spin on arrival.
        yaw = math.atan2(xy[1] - robot[1], xy[0] - robot[0])
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        self.get_logger().info(f'Heading to frontier at ({xy[0]:.2f}, {xy[1]:.2f}), {size_m:.2f} m long')
        self.goal_xy = xy
        self.goal_sent_at = self.get_clock().now()
        self.nav.send_goal_async(goal).add_done_callback(self.on_goal_response)

    def on_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Nav2 rejected the goal')
            self.blacklist.append(self.goal_xy)
            self.clear_goal()
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self.on_result(handle, f))

    def on_result(self, handle, future):
        if handle is not self.goal_handle:
            return  # a goal we already cancelled
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Reached frontier')
        else:
            self.get_logger().warn(f'Nav2 failed to reach the frontier (status {status}); skipping it')
            self.blacklist.append(self.goal_xy)
        self.clear_goal()

    def cancel_goal(self, reason, blacklist):
        self.get_logger().info(f'Abandoning frontier: {reason}')
        if blacklist:
            self.blacklist.append(self.goal_xy)
        self.goal_handle.cancel_goal_async()
        self.clear_goal()

    def clear_goal(self):
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
