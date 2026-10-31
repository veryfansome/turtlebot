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
