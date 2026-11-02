from pet_drive.safety import CLIFF, OK, WHEEL_DROP, Limits, SafetyGate


def driving_gate(now=0.0):
    """A gate with a fresh scan and command, as when everything is working."""
    gate = SafetyGate(Limits())
    gate.on_scan(now)
    return gate


def test_passes_a_safe_command():
    gate = driving_gate()
    gate.on_command(0.2, 0.5, 0.0)
    assert gate.output(0.1) == (0.2, 0.5, OK)


def test_caps_speeds():
    gate = driving_gate()
    gate.on_command(1.0, -9.0, 0.0)
    assert gate.output(0.0) == (0.3, -1.5, 'speed capped')
    gate.on_command(-1.0, 0.0, 0.0)
    assert gate.output(0.0) == (-0.15, 0.0, 'speed capped')


def test_stops_when_commands_stop():
    gate = driving_gate()
    gate.on_command(0.2, 0.0, 0.0)
    gate.on_scan(0.6)
    assert gate.output(0.6) == (0.0, 0.0, 'no command')
    assert not gate.commanded(0.6)


def test_stops_without_lidar():
    gate = SafetyGate(Limits())
    gate.on_command(0.2, 0.0, 0.0)
    assert gate.output(0.0)[2] == 'lidar silent'
    gate.on_scan(0.0)
    gate.on_command(0.2, 0.0, 1.2)
    assert gate.output(1.2) == (0.0, 0.0, 'lidar silent')


def test_stops_when_picked_up():
    gate = driving_gate()
    gate.on_command(0.2, 0.0, 0.0)
    gate.on_kidnap(True)
    assert gate.output(0.0) == (0.0, 0.0, 'picked up')
    gate.on_kidnap(False)
    assert gate.output(0.0)[2] == OK


def test_stops_on_cliff_or_wheel_drop_while_reported():
    for hazard in (CLIFF, WHEEL_DROP):
        gate = driving_gate()
        gate.on_hazards([hazard], 0.0)
        gate.on_command(-0.1, 0.0, 0.0)
        assert gate.output(0.1) == (0.0, 0.0, 'cliff or wheel drop')
        gate.on_scan(0.5)
        gate.on_command(-0.1, 0.0, 0.5)
        assert gate.output(0.5) == (-0.1, 0.0, OK)


def test_bumps_dont_change_commands():
    gate = driving_gate()
    gate.on_hazards([1], 0.0)  # BUMP
    gate.on_command(0.2, 0.0, 0.0)
    assert gate.output(0.0) == (0.2, 0.0, OK)


def test_stops_announce_and_resuming_releases_the_light_ring():
    from pet_drive.feedback import RESUMED, STOPS, announcement
    notes, colour = announcement('ok', 'lidar silent')
    assert notes == STOPS['lidar silent'][0] and colour == (255, 0, 0)
    assert announcement('lidar silent', 'ok') == (RESUMED, None)
    # Ordinary starts and ends of driving: a beep when commands stop, nothing when they start.
    assert announcement('ok', 'no command')[0] == STOPS['no command'][0]
    assert announcement('no command', 'idle') == (None, None)
    assert announcement('idle', 'ok') == (None, None)
    assert announcement('no command', 'ok') == (None, None)


def test_the_start_tone_sounds_like_no_other():
    from pet_drive.feedback import DISTRESS, RESCUE_NOTED, RESUMED, STARTED, STOPS
    others = [notes for notes, _ in STOPS.values()] + [RESUMED, RESCUE_NOTED, DISTRESS]
    assert all(STARTED != notes for notes in others)
    # The only single note, and the longest: the stop tones are short beeps or several notes.
    assert len(STARTED) == 1 and all(len(n) > 1 or n[0][1] < STARTED[0][1] for n in others)


def test_lidar_silence_off_the_dock_sounds_an_alarm_that_repeats_until_scans_return():
    from pet_drive.feedback import LidarWatch
    watch = LidarWatch(silent_after=2.0, repeat=20.0)
    assert watch.check(100.0) is None  # no scan yet: nothing to miss
    watch.on_scan(100.0)
    assert watch.check(101.5) is None
    assert watch.check(102.5) == 'silent'
    assert watch.check(110.0) is None  # once, not every check
    assert watch.check(122.5) == 'silent'  # reminded every 20 s
    events = []
    for k in range(6):  # back steadily: a scan every 0.5 s
        watch.on_scan(130.0 + k * 0.5)
        events.append(watch.check(130.0 + k * 0.5))
    assert events == [None] * 4 + ['back', None]  # after 2 s of steady scans


def test_losing_the_lidar_again_soon_after_it_came_back_sounds_at_once():
    from pet_drive.feedback import LidarWatch
    watch = LidarWatch(silent_after=2.0, repeat=20.0, steady=2.0)
    watch.on_scan(0.0)
    assert watch.check(2.5) == 'silent'
    for k in range(6):
        watch.on_scan(3.0 + k * 0.5)
        watch.check(3.0 + k * 0.5)
    assert not watch.alarm_on  # back
    assert watch.check(8.0) == 'silent'  # lost again 5.5 s after the last alarm: no 20 s wait


def test_docking_and_undocking_between_checks_doesnt_fake_a_return():
    from pet_drive.feedback import LidarWatch
    watch = LidarWatch(silent_after=2.0, startup=10.0)
    watch.on_scan(0.0)
    assert watch.check(2.5) == 'silent'
    watch.on_docked(True, 2.7)
    watch.on_docked(False, 2.9)  # both before the next check
    events = [watch.check(3.0 + k * 0.5) for k in range(10)]
    assert 'back' not in events


def test_scans_getting_through_now_and_then_dont_make_the_alarm_chatter():
    from pet_drive.feedback import LidarWatch
    watch = LidarWatch(silent_after=2.0, repeat=20.0, steady=2.0)
    events = []
    for k in range(200):  # 100 s, checked every 0.5 s, one scan in 3 s getting through
        t = k * 0.5
        if k % 6 == 0:
            watch.on_scan(t)
        events.append(watch.check(t))
    assert events.count('silent') <= 6 and events.count('back') == 0


def test_a_lidar_that_never_started_is_an_alarm_after_undocking():
    from pet_drive.feedback import LidarWatch
    watch = LidarWatch(silent_after=2.0, startup=10.0)
    watch.on_docked(True, 0.0)  # started on the dock: no scan yet
    watch.on_docked(False, 5.0)
    assert watch.check(14.5) is None and watch.check(15.5) == 'silent'


def test_the_lidar_turning_off_on_the_dock_is_not_an_alarm():
    from pet_drive.feedback import LidarWatch
    watch = LidarWatch(silent_after=2.0, startup=10.0)
    watch.on_scan(0.0)
    watch.on_docked(True, 1.0)
    assert watch.check(60.0) is None
    watch.on_docked(False, 60.0)  # undocked: the lidar takes a few seconds to spin up
    assert all(watch.check(60.0 + k * 0.5) is None for k in range(19))  # no false alarm while it starts
    watch.on_scan(64.0)
    assert watch.check(64.5) is None


def test_a_lidar_that_doesnt_restart_after_undocking_is_an_alarm():
    from pet_drive.feedback import LidarWatch
    watch = LidarWatch(silent_after=2.0, startup=10.0)
    watch.on_scan(0.0)
    watch.on_docked(True, 1.0)
    watch.on_docked(False, 60.0)
    assert watch.check(69.5) is None and watch.check(70.5) == 'silent'


def test_the_call_for_help_sounds_like_no_other():
    from pet_drive.feedback import DISTRESS, RESCUE_NOTED, RESUMED, STARTED, STOPS
    others = [notes for notes, _ in STOPS.values()] + [RESUMED, RESCUE_NOTED, STARTED]
    assert all(DISTRESS != notes for notes in others)


def drive(watch, t0, t1, forward, floor_speed, checks=None):
    """Commands at 20 Hz from t0 to t1; the floor sensor moves floor_speed m/s; checks every 0.5 s."""
    events = []
    k, x = 0, getattr(watch, '_x', 0.0)
    while t0 + k * 0.05 <= t1 + 1e-9:
        t = round(t0 + k * 0.05, 2)
        f = forward(t) if callable(forward) else forward
        watch.on_command(f, t)
        x += floor_speed * 0.05
        watch.on_floor(x, 0.0, t)
        if k % 10 == 0:
            events.append(watch.check(t))
        k += 1
    watch._x = x
    return events


def test_driving_without_getting_anywhere_is_a_call_for_help_and_repeats_while_stuck():
    from pet_drive.feedback import DistressWatch
    watch = DistressWatch(window=5.0, repeat=10.0)
    events = drive(watch, 0.0, 20.0, 0.2, 0.0)
    calls = [i * 0.5 for i, e in enumerate(events) if e == 'stuck']
    assert calls and 5.0 <= calls[0] <= 6.0  # once 5 s of commands went nowhere
    assert len(calls) == 2  # then every 10 s


def test_rocking_back_and_forth_while_stuck_is_a_call_for_help():
    from pet_drive.feedback import DistressWatch
    watch = DistressWatch()
    rocking = lambda t: 0.25 if int(t) % 2 == 0 else -0.25  # reversing every second
    assert 'stuck' in drive(watch, 0.0, 8.0, rocking, 0.02)  # the floor sensor jiggles a little


def test_a_measure_hovering_at_the_threshold_doesnt_repeat_the_call_early():
    from pet_drive.feedback import DistressWatch
    watch = DistressWatch(window=5.0, repeat=10.0)
    calls = []
    for k in range(400):  # 20 s, checked every 0.5 s
        t = round(k * 0.05, 2)
        watch.on_command(0.2, t)
        watch.on_floor((0.1 if 7.6 <= t < 8.1 else 0.057) * t, 0.0, t)  # stuck, with a brief bump at 7.6 s
        if k % 10 == 0 and watch.check(t) == 'stuck':
            calls.append(t)
    assert calls and all(b - a >= 10.0 for a, b in zip(calls, calls[1:]))


def test_driving_turning_in_circles_or_stopping_isnt_stuck():
    from pet_drive.feedback import DistressWatch
    watch = DistressWatch()
    assert 'stuck' not in drive(watch, 0.0, 10.0, 0.3, 0.28)  # moving (a little slower than told)
    assert 'stuck' not in drive(watch, 10.05, 20.0, 0.2, 0.2)  # tight circles: the path still matches
    assert 'stuck' not in drive(watch, 20.05, 30.0, 0.0, 0.0)  # not told to move


def test_floor_readings_that_stop_arriving_dont_look_like_being_stuck():
    from pet_drive.feedback import DistressWatch
    watch = DistressWatch()
    drive(watch, 0.0, 3.0, 0.2, 0.2)  # driving normally, then the floor sensor's messages stop
    events = []
    for k in range(200):
        t = round(3.05 + k * 0.05, 2)
        watch.on_command(0.2, t)
        if k % 10 == 0:
            events.append(watch.check(t))
    assert 'stuck' not in events


def test_wheels_without_power_off_the_floor_or_on_the_dock_is_not_distress():
    from pet_drive.feedback import DistressWatch
    watch = DistressWatch(unpowered_after=1.0)
    watch.on_kidnap(True, 0.0)  # picked up: the base cuts the wheels on its own
    watch.on_wheels(False, 0.0)
    assert watch.check(2.0) is None
    watch.on_kidnap(False, 2.5)  # set down: the base powers the wheels again within about 0.6 s
    assert watch.check(3.1) is None
    assert watch.check(3.5) == 'wheels without power'  # still without power on the floor: an e-stop
    watch.on_wheels(True, 3.6)
    assert watch.check(3.6) is None
    watch.on_wheels(False, 14.0)  # e-stopped again later
    assert watch.check(14.5) is None and watch.check(15.0) == 'wheels without power'
    watch.on_docked(True)
    assert watch.check(20.0) is None


def test_rescue_button_beeps_once_per_press():
    from pet_drive.feedback import RESCUE_BUTTON, rescue_pressed
    up, down = [0] * 12, [0] * 12
    down[RESCUE_BUTTON] = 1
    held, pressed = rescue_pressed(False, down)
    assert held and pressed
    held, pressed = rescue_pressed(held, down)  # still held: no second beep
    assert held and not pressed
    assert rescue_pressed(held, up) == (False, False)
    assert rescue_pressed(False, []) == (False, False)
