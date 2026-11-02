"""The safety layer: the only node that sends drive commands to the wheels.

Whoever drives (a scripted wander, a learned policy) publishes the speeds it wants on drive/cmd_vel;
this node applies the rules in safety.py and sends the result to cmd_vel at a steady rate. When
nobody is driving it goes quiet, so the controller can still drive the robot directly.
"""
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import TwistStamped
from irobot_create_msgs.msg import (AudioNote, AudioNoteVector, DockStatus, HazardDetectionVector, KidnapStatus, LedColor,
                                    LightringLeds, Mouse, WheelStatus)
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String

from pet_drive.feedback import (DISTRESS, RESCUE_NOTED, RESUMED, STARTED, STOPS, DistressWatch, LidarWatch, announcement,
                                rescue_pressed)
from pet_drive.safety import Limits, SafetyGate

# Zero commands sent after the driver stops, so the robot stops even if one is lost.
STOP_REPEATS = 3
# Half-second checks the base's audio relay must be subscribed for before the start tone plays.
START_TONE_WAIT = 4


class SafetyNode(Node):

    def __init__(self):
        super().__init__('drive_safety')
        defaults = Limits()
        self.gate = SafetyGate(Limits(**{
            name: self.declare_parameter(name, getattr(defaults, name)).value
            for name in vars(defaults)
        }))
        rate = self.declare_parameter('rate', 20.0).value  # Hz
        # Tones and light ring colours when it stops, so the person nearby knows why.
        self.feedback = self.declare_parameter('feedback', True).value
        self.stops_left = 0
        self.reason = None
        self.ring_overridden = False
        self.rescue_held = False
        self.lidar_watch = LidarWatch()
        self.distress = DistressWatch()

        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        # Why the last command was changed ('ok' if it wasn't), for logs and the recorder.
        self.status_pub = self.create_publisher(String, 'drive/safety', 10)
        self.audio_pub = self.create_publisher(AudioNoteVector, 'cmd_audio', 10)
        self.ring_pub = self.create_publisher(LightringLeds, 'cmd_lightring', 10)
        # Calls for help (DistressWatch), for whoever's nearby and for the recorder.
        self.distress_pub = self.create_publisher(String, 'drive/distress', 10)
        self.create_subscription(TwistStamped, 'drive/cmd_vel', self.on_command, 10)
        self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(HazardDetectionVector, 'hazard_detection', self.on_hazards, qos_profile_sensor_data)
        self.create_subscription(KidnapStatus, 'kidnap_status', self.on_kidnap, qos_profile_sensor_data)
        self.create_subscription(Joy, 'joy', self.on_joy, qos_profile_sensor_data)
        self.create_subscription(DockStatus, 'dock_status', self.on_dock, qos_profile_sensor_data)
        self.create_timer(0.5, self.check_lidar)
        # What the wheels were told (by this node or the controller), the floor sensor and motor power.
        self.create_subscription(TwistStamped, 'cmd_vel', self.on_wheel_command, 10)
        self.create_subscription(Mouse, 'mouse', self.on_mouse, qos_profile_sensor_data)
        self.create_subscription(WheelStatus, 'wheel_status', self.on_wheel_status, qos_profile_sensor_data)
        self.create_timer(0.5, self.check_distress)
        self.create_timer(1.0 / rate, self.tick)
        if self.feedback:
            self.listener_seen = 0
            self.start_timer = self.create_timer(0.5, self.announce_start)
        self.get_logger().info(f'Safety layer up: {self.gate.limits}')

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def on_command(self, msg: TwistStamped):
        self.gate.on_command(msg.twist.linear.x, msg.twist.angular.z, self.now())

    def on_scan(self, msg: LaserScan):
        self.gate.on_scan(self.now())
        self.lidar_watch.on_scan(self.now())

    def on_dock(self, msg: DockStatus):
        self.lidar_watch.on_docked(msg.is_docked, self.now())
        self.distress.on_docked(msg.is_docked)

    def check_lidar(self):
        event = self.lidar_watch.check(self.now())
        # While a driver is driving, the safety layer's own lidar-silent stop already plays these.
        audible = self.feedback and self.reason in (None, 'idle')
        if event == 'silent':
            self.get_logger().warn(f'Lidar silent for {self.now() - self.lidar_watch.last_scan:.0f} s while undocked')
            if audible:
                self.play(STOPS['lidar silent'][0], queue=True)
        elif event == 'back':
            self.get_logger().info('Lidar back')
            if audible:
                self.play(RESUMED, queue=True)

    def on_hazards(self, msg: HazardDetectionVector):
        self.gate.on_hazards([d.type for d in msg.detections], self.now())

    def on_kidnap(self, msg: KidnapStatus):
        self.gate.on_kidnap(msg.is_kidnapped)
        self.distress.on_kidnap(msg.is_kidnapped, self.now())

    def on_wheel_command(self, msg: TwistStamped):
        self.distress.on_command(msg.twist.linear.x, self.now())

    def on_mouse(self, msg: Mouse):
        self.distress.on_floor(msg.integrated_x, msg.integrated_y, self.now())

    def on_wheel_status(self, msg: WheelStatus):
        self.distress.on_wheels(msg.wheels_enabled, self.now())

    def check_distress(self):
        why = self.distress.check(self.now())
        if why is None:
            return
        self.get_logger().warn(f'Calling for help: {why}')
        self.distress_pub.publish(String(data=why))
        if self.feedback:
            self.play(DISTRESS, queue=True)

    def announce_start(self):
        # Waits for the relay to the base to subscribe, or the tone would go nowhere, and then a
        # little longer: a subscriber can be seen before messages actually reach it.
        if self.audio_pub.get_subscription_count() == 0:
            self.listener_seen = 0
            return
        self.listener_seen += 1
        if self.listener_seen >= START_TONE_WAIT:
            self.play(STARTED)
            self.get_logger().info('Start tone played')
            self.start_timer.cancel()

    def on_joy(self, msg: Joy):
        self.rescue_held, pressed = rescue_pressed(self.rescue_held, msg.buttons)
        if pressed:
            self.get_logger().info('Rescue button pressed')
            if self.feedback:
                self.play(RESCUE_NOTED)

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
            if self.feedback and self.reason is not None:
                self.announce(*announcement(self.reason, reason))
            self.reason = reason
            # rclpy refuses to log from one line at different severities, so two calls.
            if reason in ('ok', 'idle', 'no command', 'speed capped'):
                self.get_logger().info(f'Drive: {reason}')
            else:
                self.get_logger().warn(f'Drive: {reason}')

    def play(self, notes, queue=False):
        """Tones only: the light ring stays as it is (a stop's colour, say). queue: after whatever is
        playing, rather than cutting it off (for alarms that can coincide, so neither drowns the other)."""
        audio = AudioNoteVector(append=queue)
        for frequency, seconds in notes:
            audio.notes.append(AudioNote(
                frequency=frequency, max_runtime=Duration(sec=int(seconds), nanosec=int(seconds % 1 * 1e9))))
        self.audio_pub.publish(audio)

    def announce(self, notes, colour):
        """A stop's tones and colour (colour None gives the light ring back to the robot)."""
        if notes:
            self.play(notes)
        if colour is not None:
            ring = LightringLeds(override_system=True)
            ring.leds = [LedColor(red=colour[0], green=colour[1], blue=colour[2]) for _ in range(6)]
            self.ring_pub.publish(ring)
            self.ring_overridden = True
        elif self.ring_overridden:
            self.ring_pub.publish(LightringLeds(override_system=False))  # give the ring back to the robot
            self.ring_overridden = False

    def stop(self):
        self.send(0.0, 0.0)
        if self.ring_overridden:
            self.ring_pub.publish(LightringLeds(override_system=False))


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
