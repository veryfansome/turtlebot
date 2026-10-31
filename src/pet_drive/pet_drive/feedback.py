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


def announcement(previous: str, current: str):
    """What to play and show when the safety layer's reason changes from previous to current:
    (notes or None, colour, or None to give the light ring back to the robot)."""
    if current in STOPS:
        notes, colour = STOPS[current]
        return notes, colour
    resumed = previous in STOPS and previous != 'no command' and current == 'ok'
    return (RESUMED if resumed else None), None
