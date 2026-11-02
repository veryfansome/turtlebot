"""How the safety layer tells the person nearby why it stopped: a tone pattern on the base's speaker
and a colour on its light ring. The base can't speak, only play tones.

Kept free of ROS. Notes are (frequency in Hz, seconds); colours are (red, green, blue), 0-255.
"""
import math

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


# A call for help (DistressWatch): a two-note siren, unlike any stop tone.
DISTRESS = [(587, 0.25), (440, 0.25)] * 2


class DistressWatch:
    """Notices when the robot needs a person, so it can call for help (and recordings get the label
    without anyone pressing a button): stuck, or its wheels without power for unpowered_after seconds
    on the floor (an e-stop). Being picked up or docked isn't distress. check() says when to call: when
    it starts, then every repeat seconds while it lasts.

    Stuck means told to travel at least min_commanded metres over the last window seconds, while the
    floor sensor measured less than ratio of that. Checked against the recordings (2026-11-01): when
    stuck, a person rocks it back and forth, so the commands keep reversing and the floor sensor still
    jiggles a few centimetres; travel against commands caught all 8 rescues (with 17 other calls in 82
    minutes, many of them likely real struggles that ended without help). The floor sensor measures
    path travelled, so tight circles don't look stuck."""

    def __init__(self, window: float = 5.0, min_commanded: float = 0.5, ratio: float = 0.3,
                 unpowered_after: float = 1.0, repeat: float = 10.0, command_timeout: float = 0.5):
        self.window, self.min_commanded, self.ratio = window, min_commanded, ratio
        self.unpowered_after, self.repeat, self.command_timeout = unpowered_after, repeat, command_timeout
        self.commanded = self.travelled = 0.0  # running totals (m)
        self.last_command = None  # (t, forward)
        self.last_floor = None  # (x, y)
        self.floor_at = None  # its latest reading: without fresh ones, travel can't be told
        self.history = []  # (t, commanded, travelled) at each check
        self.unpowered_since = None
        self.kidnapped = self.docked = False
        self.called, self.called_at = None, None  # the last call: its reason and when

    def on_command(self, forward: float, t: float):
        if self.last_command is not None:  # each command holds until the next, for up to command_timeout
            t0, f0 = self.last_command
            self.commanded += abs(f0) * min(max(t - t0, 0.0), self.command_timeout)
        self.last_command = (t, forward)

    def on_floor(self, x: float, y: float, t: float):
        """The floor sensor's integrated position (m)."""
        if self.last_floor is not None:
            self.travelled += math.hypot(x - self.last_floor[0], y - self.last_floor[1])
        self.last_floor, self.floor_at = (x, y), t

    def on_wheels(self, enabled: bool, t: float):
        if enabled:
            self.unpowered_since = None
        elif self.unpowered_since is None:
            self.unpowered_since = t

    def on_kidnap(self, kidnapped: bool, t: float):
        if self.kidnapped and not kidnapped and self.unpowered_since is not None:
            self.unpowered_since = t  # set down: the base takes about 0.6 s to power the wheels again
        self.kidnapped = kidnapped

    def on_docked(self, docked: bool):
        self.docked = docked

    def reason(self, t: float):
        """'wheels without power', 'stuck' or None."""
        if self.kidnapped or self.docked:
            self.history.clear()
            return None
        if self.unpowered_since is not None:
            self.history.clear()  # travel without power isn't the robot's doing
            return 'wheels without power' if t - self.unpowered_since >= self.unpowered_after else None
        if self.floor_at is None or t - self.floor_at > self.command_timeout:
            self.history.clear()
            return None
        self.history.append((t, self.commanded, self.travelled))
        while len(self.history) > 1 and self.history[1][0] <= t - self.window:
            self.history.pop(0)
        t0, c0, f0 = self.history[0]
        if t - t0 < self.window:
            return None
        commanded, travelled = self.commanded - c0, self.travelled - f0
        if commanded >= self.min_commanded and travelled < self.ratio * commanded:
            return 'stuck'
        return None

    def check(self, t: float):
        """The reason to call for help now, or None. Call it regularly (it keeps the travel window). The
        same reason is called at most every repeat seconds, even if it briefly went away (a measure
        hovering at the threshold); a different reason is called at once."""
        why = self.reason(t)
        if why is None:
            return None
        if why != self.called or t - self.called_at >= self.repeat:
            self.called, self.called_at = why, t
            return why
        return None


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
