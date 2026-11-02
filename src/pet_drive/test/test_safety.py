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
    from pet_drive.feedback import RESCUE_NOTED, RESUMED, STARTED, STOPS
    others = [notes for notes, _ in STOPS.values()] + [RESUMED, RESCUE_NOTED]
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
