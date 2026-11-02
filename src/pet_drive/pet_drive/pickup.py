"""When the robot is picked up and set down somewhere, SLAM's map no longer matches where it is: SLAM
tracks it mostly by wheel motion, and the wheels didn't see the carry. So: pause
SLAM while it's in the air (the view from mid-air shouldn't go into the map), and once it has been set
down and settled, reset SLAM to a fresh map, as if a new session started.

Kept free of ROS: PickupTracker decides when; pickup_node does it.
"""


class PickupTracker:
    """Follows the base's kidnap status. on_kidnap() says 'lifted' when the robot is picked up;
    reset_due() says True once, settle seconds after it's set down (a person often lifts it twice in
    quick succession; a new lift before then cancels the pending reset)."""

    def __init__(self, settle: float = 1.0):
        self.settle = settle
        self.lifted = False
        self.landed_at = None  # set down, reset pending
        self.pickups = 0

    def on_kidnap(self, kidnapped: bool, t: float):
        if kidnapped and not self.lifted:
            self.lifted = True
            self.landed_at = None
            self.pickups += 1
            return 'lifted'
        if not kidnapped and self.lifted:
            self.lifted = False
            self.landed_at = t
            return 'landed'
        return None

    def reset_due(self, t: float) -> bool:
        if self.landed_at is not None and t - self.landed_at >= self.settle:
            self.landed_at = None
            return True
        return False
