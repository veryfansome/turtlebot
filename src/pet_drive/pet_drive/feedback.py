"""How the safety layer tells the person nearby why it stopped: a tone pattern on the base's speaker
and a colour on its light ring. The base can't speak, only play tones.

Kept free of ROS. Notes are (frequency in Hz, seconds); colours are (red, green, blue), 0-255.
"""

# Why the safety layer stopped the robot → what it plays and shows while stopped.
STOPS = {
    'no command': ([(440, 0.15)], (255, 140, 0)),  # one short beep; amber
    'lidar silent': ([(880, 0.2), (660, 0.2), (440, 0.4)], (255, 0, 0)),  # falling; red
    'cliff or wheel drop': ([(1760, 0.1), (1320, 0.1)] * 3, (255, 0, 255)),  # fast warble; magenta
    'picked up': ([(523, 0.1), (659, 0.1), (784, 0.2)], (0, 80, 255)),  # rising; blue
}
# Played when driving resumes after one of the stops above (not after 'no command': that's just
# someone starting to drive).
RESUMED = [(660, 0.1), (880, 0.1)]

# The controller's rescue button: pressed just before a person frees a stuck robot, so recordings
# mark it (learn/pet_learn/convert.py reads the same button). Y on the TurtleBot 4 controller; X is
# the display menu's Back and B the e-stop. The tone confirms the press was seen.
RESCUE_BUTTON = 2
RESCUE_NOTED = [(392, 0.15), (392, 0.15)]  # two low beeps
# Played once when the safety layer starts and the base is listening: a recording has begun.
STARTED = [(587, 0.5)]  # one long note: unlike every stop tone, which are short or several


class LidarWatch:
    """Notices the lidar's scans stopping while nobody is driving (the safety layer's stop only covers a
    driver), so a person recording knows at once instead of driving on without it. The lidar turns off
    on the dock, so silence there doesn't count, and after undocking it gets startup seconds to spin
    up (about 4 s on the robot). check() says when to sound the alarm: once scans have been silent for
    silent_after seconds, then at most every repeat seconds while they're missing, and once when
    they've been back steadily for steady seconds: a scan in the last fresh seconds at every check
    (scans getting through now and then over a bad link shouldn't make it chatter)."""

    def __init__(self, silent_after: float = 2.0, repeat: float = 20.0, startup: float = 10.0, steady: float = 2.0,
                 fresh: float = 0.5):
        self.silent_after, self.repeat, self.startup, self.steady = silent_after, repeat, startup, steady
        self.fresh = fresh
        self.last_scan = None  # none yet: nothing to miss
        self.docked = False
        self.alarm_on = False
        self.alarmed_at = None  # when the alarm last sounded
        self.back_since = None  # scans arriving again since, while the alarm is on

    def on_scan(self, t: float):
        self.last_scan = t

    def on_docked(self, docked: bool, t: float):
        if docked and not self.docked:  # the lidar turns off on the dock: that ends any alarm
            self.alarm_on, self.back_since = False, None
        if self.docked and not docked:  # undocked: the lidar is starting up, and should be within startup s
            self.last_scan = max(self.last_scan if self.last_scan is not None else t, t + self.startup - self.silent_after)
        self.docked = docked

    def check(self, t: float):
        """'silent' (sound the alarm), 'back' (scans returned after an alarm) or None."""
        silent = (self.last_scan is not None and not self.docked and t - self.last_scan > self.silent_after)
        if silent:
            self.back_since = None
            if self.alarmed_at is None or t - self.alarmed_at >= self.repeat:
                self.alarm_on, self.alarmed_at = True, t
                return 'silent'
            return None
        if not self.alarm_on:
            return None
        if self.docked:  # docked since: the lidar is off on purpose
            self.alarm_on = False
            return None
        # Not at the lidar's normal pace (about 7 a second), or only the startup allowance after undocking.
        if t - self.last_scan > self.fresh or self.last_scan > t:
            self.back_since = None
            return None
        self.back_since = t if self.back_since is None else self.back_since
        if t - self.back_since >= self.steady:
            self.alarm_on, self.alarmed_at, self.back_since = False, None, None  # a new loss sounds at once
            return 'back'
        return None


def rescue_pressed(was_held: bool, buttons) -> tuple:
    """(held now, pressed just now) from a controller message's buttons."""
    held = len(buttons) > RESCUE_BUTTON and buttons[RESCUE_BUTTON] == 1
    return held, held and not was_held


def announcement(previous: str, current: str):
    """What to play and show when the safety layer's reason changes from previous to current:
    (notes or None, colour, or None to give the light ring back to the robot)."""
    if current in STOPS:
        notes, colour = STOPS[current]
        return notes, colour
    resumed = previous in STOPS and previous != 'no command' and current == 'ok'
    return (RESUMED if resumed else None), None
