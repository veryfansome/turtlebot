from pet_drive.pickup import PickupTracker


def test_slam_resets_once_the_robot_has_settled_after_being_set_down():
    tracker = PickupTracker(settle=1.0)
    assert tracker.on_kidnap(False, 0.0) is None
    assert tracker.on_kidnap(True, 1.0) == 'lifted' and tracker.pickups == 1
    assert tracker.on_kidnap(True, 1.1) is None  # repeated status: still lifted
    assert not tracker.reset_due(2.0)  # in the air: no reset
    assert tracker.on_kidnap(False, 3.0) == 'landed'
    assert not tracker.reset_due(3.5)
    assert tracker.reset_due(4.0)
    assert not tracker.reset_due(5.0)  # once


def test_a_second_lift_before_it_settles_waits_for_the_last_set_down():
    tracker = PickupTracker(settle=1.0)
    tracker.on_kidnap(True, 0.0)
    tracker.on_kidnap(False, 1.0)
    assert tracker.on_kidnap(True, 1.5) == 'lifted'  # picked up again
    assert not tracker.reset_due(2.5)
    tracker.on_kidnap(False, 3.0)
    assert not tracker.reset_due(3.9) and tracker.reset_due(4.0)
    assert tracker.pickups == 2
