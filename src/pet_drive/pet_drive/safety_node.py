"""The safety layer: the only node that sends drive commands to the wheels.

Whoever drives (a scripted wander, a learned policy) publishes the speeds it wants on drive/cmd_vel;
this node applies the rules in safety.py and sends the result to cmd_vel at a steady rate. When
nobody is driving it goes quiet, so the controller can still drive the robot directly.
"""
from geometry_msgs.msg import TwistStamped
from irobot_create_msgs.msg import HazardDetectionVector, KidnapStatus
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from pet_drive.safety import Limits, SafetyGate

# Zero commands sent after the driver stops, so the robot stops even if one is lost.
STOP_REPEATS = 3


class SafetyNode(Node):

    def __init__(self):
        super().__init__('drive_safety')
        defaults = Limits()
        self.gate = SafetyGate(Limits(**{
            name: self.declare_parameter(name, getattr(defaults, name)).value
            for name in vars(defaults)
        }))
        rate = self.declare_parameter('rate', 20.0).value  # Hz
        self.stops_left = 0
        self.reason = None

        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        # Why the last command was changed ('ok' if it wasn't), for logs and the recorder.
        self.status_pub = self.create_publisher(String, 'drive/safety', 10)
        self.create_subscription(TwistStamped, 'drive/cmd_vel', self.on_command, 10)
        self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(HazardDetectionVector, 'hazard_detection', self.on_hazards, qos_profile_sensor_data)
        self.create_subscription(KidnapStatus, 'kidnap_status', self.on_kidnap, qos_profile_sensor_data)
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f'Safety layer up: {self.gate.limits}')

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def on_command(self, msg: TwistStamped):
        self.gate.on_command(msg.twist.linear.x, msg.twist.angular.z, self.now())

    def on_scan(self, msg: LaserScan):
        self.gate.on_scan(self.now())

    def on_hazards(self, msg: HazardDetectionVector):
        self.gate.on_hazards([d.type for d in msg.detections], self.now())

    def on_kidnap(self, msg: KidnapStatus):
        self.gate.on_kidnap(msg.is_kidnapped)

    def tick(self):
        now = self.now()
        if self.gate.commanded(now):
            self.stops_left = STOP_REPEATS
        elif self.stops_left > 0:
            self.stops_left -= 1
        else:
            self.set_reason('idle')
            return  # nobody driving: stay quiet so the controller can drive
        forward, turn, reason = self.gate.output(now)
        self.send(forward, turn)
        self.status_pub.publish(String(data=reason))
        self.set_reason(reason)

    def send(self, forward, turn):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(forward)
        msg.twist.angular.z = float(turn)
        self.cmd_pub.publish(msg)

    def set_reason(self, reason):
        if reason != self.reason:
            self.reason = reason
            # rclpy refuses to log from one line at different severities, so two calls.
            if reason in ('ok', 'idle', 'no command', 'speed capped'):
                self.get_logger().info(f'Drive: {reason}')
            else:
                self.get_logger().warn(f'Drive: {reason}')

    def stop(self):
        self.send(0.0, 0.0)


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
