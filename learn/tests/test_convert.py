import numpy as np

from pet_learn.convert import SCAN_SECTORS, build_steps


def series(times_s, values):
    return np.asarray(np.array(times_s) * 1e9, dtype=np.int64), list(values)


def test_steps_continue_after_the_scans_stop_with_the_scan_marked_missing():
    names = ('scan', 'cmd', 'joy', 'odom', 'wheel_vels', 'wheel_status', 'mouse', 'slip', 'ir', 'kidnap', 'dock',
             'battery', 'rescue', 'map_odom', 'odom_base', 'map', 'hazard')
    s = {name: series([], []) for name in names}
    scan_t = np.arange(0, 2.0, 0.1)  # scans for 2 s ...
    odom_t = np.arange(0, 5.0, 0.05)  # ... odometry for 5 s
    s['scan'] = series(scan_t, [np.full(SCAN_SECTORS, 1.0, np.float32)] * len(scan_t))
    s['odom'] = series(odom_t, [(0.2, 0.0)] * len(odom_t))
    s['wheel_status'] = series(odom_t[odom_t < 3.0], [(100, 100)] * int((odom_t < 3.0).sum()))  # stops at 3 s
    s['map_odom'] = series([0.0], [(0.0, 0.0, 0.0)])
    s['odom_base'] = series(odom_t, [(0.2 * t, 0.0, 0.0) for t in odom_t])
    steps = build_steps(s)
    t, age = steps['t'], steps['scan_age']
    assert t[-1] >= 4.9
    assert np.isnan(steps['scan'][t > 2.5]).all() and np.isfinite(steps['scan'][t < 1.9]).all()
    assert np.all(age[t < 1.9] <= 0.1 + 1e-6) and np.all(age[t > 2.5] > 0.5)
    assert np.isfinite(steps['odom_vel']).all()
    assert np.isnan(steps['wheel_current'][t > 3.6]).all() and np.isfinite(steps['wheel_current'][t < 2.9]).all()
    assert np.isfinite(steps['pose'][t > 2.5]).all()  # SLAM's last correction carries on with odometry
