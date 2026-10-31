"""The safety layer's rules: what a proposed (forward speed, turn rate) becomes before it reaches the wheels.

Kept free of ROS so it can be tested anywhere. Times are in seconds, speeds in m/s and rad/s.
"""
from dataclasses import dataclass

# irobot_create_msgs/msg/HazardDetection types
BUMP, CLIFF, WHEEL_DROP = 1, 2, 4

OK = 'ok'


@dataclass
class Limits:
    max_forward: float = 0.3
    max_reverse: float = 0.15
    max_turn: float = 1.5
    # Stop if no command has arrived for this long: whoever was driving has stopped or crashed.
    command_timeout: float = 0.5
    # Stop if no lidar scan has arrived for this long: the driver is blind.
    scan_timeout: float = 1.0
    # After a new bump, stop for this long before accepting commands again.
    bump_pause: float = 0.5
    # The base reports hazards only while they last, so one counts as over once it hasn't been
    # reported for this long.
    hazard_hold: float = 0.3


class SafetyGate:
    """Holds what the robot last sensed and turns proposed commands into safe ones."""

    def __init__(self, limits: Limits = Limits()):
        self.limits = limits
        self.command = (0.0, 0.0)
        self.command_at = None
        self.scan_at = None
        self.kidnapped = False
        self.bump_seen_at = None  # last time a bump was reported
        self.bump_started_at = None  # when the current bump began
        self.fall_seen_at = None  # last time a cliff or wheel drop was reported

    def on_command(self, forward: float, turn: float, now: float):
        self.command = (forward, turn)
        self.command_at = now

    def on_scan(self, now: float):
        self.scan_at = now

    def on_kidnap(self, kidnapped: bool):
        self.kidnapped = kidnapped

    def on_hazards(self, types, now: float):
        """types: the hazard types in one hazard_detection message."""
        if BUMP in types:
            if not self.bumping(now):
                self.bump_started_at = now
            self.bump_seen_at = now
        if CLIFF in types or WHEEL_DROP in types:
            self.fall_seen_at = now

    def bumping(self, now: float) -> bool:
        return self.bump_seen_at is not None and now - self.bump_seen_at <= self.limits.hazard_hold

    def commanded(self, now: float) -> bool:
        """Whether someone is driving through the gate: a command arrived recently."""
        return self.command_at is not None and now - self.command_at <= self.limits.command_timeout

    def output(self, now: float) -> tuple[float, float, str]:
        """The command to send now, and why it differs from the proposed one ('ok' if it doesn't)."""
        lim = self.limits
        if self.kidnapped:
            return 0.0, 0.0, 'picked up'
        if self.fall_seen_at is not None and now - self.fall_seen_at <= lim.hazard_hold:
            return 0.0, 0.0, 'cliff or wheel drop'
        if self.scan_at is None or now - self.scan_at > lim.scan_timeout:
            return 0.0, 0.0, 'lidar silent'
        if not self.commanded(now):
            return 0.0, 0.0, 'no command'
        if self.bumping(now) and now - self.bump_started_at < lim.bump_pause:
            return 0.0, 0.0, 'bumped'
        forward, turn = self.command
        reason = OK
        if self.bumping(now) and forward > 0:
            forward, reason = 0.0, 'bumper pressed: no forward'
        clamped_forward = min(max(forward, -lim.max_reverse), lim.max_forward)
        clamped_turn = min(max(turn, -lim.max_turn), lim.max_turn)
        if reason == OK and (clamped_forward, clamped_turn) != (forward, turn):
            reason = 'speed capped'
        return clamped_forward, clamped_turn, reason
