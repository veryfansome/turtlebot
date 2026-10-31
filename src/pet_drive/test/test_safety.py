from pet_drive.safety import BUMP, CLIFF, OK, WHEEL_DROP, Limits, SafetyGate


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


def test_bump_pauses_then_allows_only_moves_away():
    gate = driving_gate()
    gate.on_hazards([BUMP], 0.0)
    gate.on_command(0.2, 0.0, 0.1)
    assert gate.output(0.1) == (0.0, 0.0, 'bumped')
    # Still pressed after the pause: no forward, but backing up and turning go through.
    gate.on_scan(0.6)
    for t in (0.2, 0.4, 0.6):
        gate.on_hazards([BUMP], t)
    gate.on_command(0.2, 1.0, 0.6)
    assert gate.output(0.6) == (0.0, 1.0, 'bumper pressed: no forward')
    gate.on_command(-0.1, 0.0, 0.6)
    assert gate.output(0.6) == (-0.1, 0.0, OK)


def test_released_bumper_allows_forward_again():
    gate = driving_gate()
    gate.on_hazards([BUMP], 0.0)
    gate.on_scan(1.0)
    gate.on_command(0.2, 0.0, 1.0)
    assert gate.output(1.0) == (0.2, 0.0, OK)


def test_a_held_bumper_is_one_bump():
    gate = driving_gate()
    for t in (0.0, 0.2, 0.4, 0.6):
        gate.on_hazards([BUMP], t)
    assert gate.bump_started_at == 0.0
    gate.on_command(-0.1, 0.0, 0.6)
    assert gate.output(0.6)[2] == OK
