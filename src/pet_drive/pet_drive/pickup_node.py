"""Resets SLAM to a fresh map after the robot is picked up and set down (see pickup.py).

Runs alongside slam_toolbox in a recording. Each reset is announced on drive/slam_reset, which the
recorder keeps, so training knows positions before and after it are on different maps.
"""
from irobot_create_msgs.msg import KidnapStatus
from rcl_interfaces.srv import GetParameters
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from slam_toolbox.srv import Pause, Reset
from std_msgs.msg import String

from pet_drive.pickup import PickupTracker


class PickupNode(Node):

    def __init__(self):
        super().__init__('pickup')
        self.tracker = PickupTracker(self.declare_parameter('settle', 1.0).value)
        # SLAM should be paused from a lift until the reset after the set-down. sync_pause() makes it
        # so, one request at a time, deciding from this when each reply comes back (a person can lift
        # the robot again while a reset or resume is still under way).
        self.want_paused = False
        self.syncing = False
        self.toggles = 0  # in a row without reaching want_paused: a cap, in case the state reads stale
        self.unsynced = False  # a sync failed (service not up yet, a failed call): tick() retries it
        self.last_retry = 0.0
        self.event_pub = self.create_publisher(String, 'drive/slam_reset', 10)
        self.slam_params = self.create_client(GetParameters, 'slam_toolbox/get_parameters')
        self.slam_pause = self.create_client(Pause, 'slam_toolbox/pause_new_measurements')
        self.slam_reset = self.create_client(Reset, 'slam_toolbox/reset')
        self.create_subscription(KidnapStatus, 'kidnap_status', self.on_kidnap, qos_profile_sensor_data)
        self.create_timer(0.2, self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def on_kidnap(self, msg: KidnapStatus):
        event = self.tracker.on_kidnap(msg.is_kidnapped, self.now())
        if event == 'lifted':
            self.get_logger().warn(f'Picked up (#{self.tracker.pickups}): pausing SLAM')
            self.want_paused = True
            self.sync_pause()
        elif event == 'landed':
            self.get_logger().info('Set down: resetting SLAM once it settles')

    def tick(self):
        if self.unsynced and not self.syncing and self.now() - self.last_retry >= 1.0:
            self.last_retry = self.now()
            self.sync_pause()
        if not self.tracker.reset_due(self.now()):
            return
        if not self.slam_reset.service_is_ready():
            self.get_logger().error('slam_toolbox/reset not available; carrying on with the old map')
            self.resume_unless_lifted()
            return
        pickup = self.tracker.pickups
        self.slam_reset.call_async(Reset.Request(pause_new_measurements=False)).add_done_callback(
            lambda f: self.on_reset(f, pickup))

    def on_reset(self, future, pickup):
        result = future.result()
        if result is None or result.result != Reset.Response.RESULT_SUCCESS:
            self.get_logger().error('SLAM reset failed; carrying on with the old map')
        else:
            # Announced even if it has been lifted again since: the map did change here.
            self.get_logger().info(f'SLAM reset after pickup #{pickup}: a fresh map')
            self.event_pub.publish(String(data=f'pickup {pickup}'))
        self.resume_unless_lifted()  # reset doesn't resume a paused SLAM, whatever it's asked

    def resume_unless_lifted(self):
        if not self.tracker.lifted:  # lifted again: stays paused until the next set-down's reset
            self.want_paused = False
            self.sync_pause()

    def sync_pause(self):
        """Bring SLAM's pause in line with want_paused. Its pause service toggles, so this reads the current
        state first, and after any change reads again, in case want_paused changed meanwhile."""
        if self.syncing:
            return  # the request under way checks want_paused again when it's done
        if not self.slam_params.service_is_ready():
            if not self.unsynced:
                self.get_logger().error('slam_toolbox not available; SLAM pause not changed yet')
            self.unsynced = True
            return
        self.syncing = True
        self.slam_params.call_async(GetParameters.Request(names=['paused_new_measurements'])).add_done_callback(
            self.on_pause_state)

    def on_pause_state(self, future):
        values = future.result().values if future.result() else []
        if not values or values[0].type != 1:  # 1: bool
            self.syncing, self.unsynced = False, True
            self.get_logger().error("Can't read SLAM's paused_new_measurements; not toggling it yet")
            return
        if values[0].bool_value == self.want_paused:
            self.syncing, self.toggles, self.unsynced = False, 0, False
            return
        if self.toggles >= 3:
            self.syncing, self.toggles, self.unsynced = False, 0, False
            self.get_logger().error("SLAM's pause state isn't following its toggles; leaving it")
            return
        self.toggles += 1
        word = 'paused' if self.want_paused else 'resumed'
        self.slam_pause.call_async(Pause.Request()).add_done_callback(lambda f: self.on_toggled(f, word))

    def on_toggled(self, future, word):
        self.syncing = False
        if future.result() and future.result().status:
            self.get_logger().info(f'SLAM {word}')
            self.sync_pause()  # check again: want_paused may have changed meanwhile
        else:
            self.unsynced = True
            self.get_logger().error(f'SLAM toggle to {word} failed; retrying')


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = PickupNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
