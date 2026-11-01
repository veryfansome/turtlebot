"""Turn recorded driving sessions into training steps at a fixed rate.

    uv run python -m pet_learn.convert [--drives ../data/drives] [--out ../data/steps] [--force]

Converts every session in drives/ that doesn't have a steps file yet (all of them with --force), and
prints a summary of each. Writes <session>.npz (the steps) and <session>.maps.npz (every SLAM map). A session is a bag directory made by `ops/field-test.sh start record`.

Each step holds what the robot sensed and was told at that moment, aligned on one clock (the time
the VM received each message). See STEP_FIELDS for what's in a steps file.
"""
import argparse
import glob
import math
import os

from mcap_ros2.reader import read_ros2_messages
import numpy as np

from pet_learn.steps import bump_side, compose, latest_before, robot_sectors, yaw_from_quaternion

RATE = 10.0  # steps per second
SCAN_SECTORS = 60
COMMAND_TIMEOUT = 0.5  # a drive command older than this no longer counts (the base stops too)
# The base's front infrared proximity sensors, left to right (their angles in the robot's TF: 75, 50,
# 25, 0, -25, -50, -75 degrees), so mirroring left to right reverses them.
# Left to right: 75°, 50°, 25° left, straight ahead (front_center_left), 25°, 50°, 75° right (from the base's TF).
IR_FRAMES = ['ir_intensity_side_left', 'ir_intensity_left', 'ir_intensity_front_left', 'ir_intensity_front_center_left',
             'ir_intensity_front_center_right', 'ir_intensity_front_right', 'ir_intensity_right']
RESCUE_BUTTON = 2  # Y on the TurtleBot 4 controller (X is the display menu's Back, B the e-stop)
HAZARD_TYPES = {'bump': 1, 'cliff': 2, 'stall': 3, 'wheel_drop': 4}

STEP_FIELDS = {
    't': 'seconds since the first scan',
    'scan': f'lidar, nearest return in each of {SCAN_SECTORS} equal sectors of the robot frame (m): sector 0 starts directly behind (-180 degrees), counterclockwise, straight ahead between sectors {SCAN_SECTORS // 2 - 1} and {SCAN_SECTORS // 2}',
    'cmd': 'drive command the wheels were told: forward m/s, turn rad/s (0 if none in the last 0.5 s)',
    'joy_active': 'a controller button (the enable button, while driving) was held: a person was driving',
    'rescue': 'the rescue button (Y) was pressed during this step: the robot was stuck and a person is about to free it',
    'odom_vel': 'wheel odometry: forward m/s, turn rad/s',
    'wheel_vels': 'wheel speeds: left, right (rad/s)',
    'wheel_current': 'motor currents: left, right (mA)',
    'mouse_speed': 'speed over the floor from the optical floor sensor (m/s)',
    'mouse_quality': 'floor sensor surface quality (higher is better tracking)',
    'slipping': "the base's wheel-slip detection",
    'ir': 'front infrared proximity readings, left to right (IR_FRAMES; higher is closer); nan if not recorded',
    'bump': 'a bump began during this step: left, centre, right',
    'stall': 'the base reported a wheel stall during this step',
    'cliff_or_drop': 'a cliff or wheel drop was reported during this step',
    'kidnapped': 'picked up',
    'docked': 'on the dock',
    'battery': 'battery fraction',
    'pose': 'SLAM pose in the map: x, y (m), yaw (rad); nan before SLAM has one',
    'known_cells': 'mapped cells (free or occupied) in the latest SLAM map',
    'new_cells': 'cells that became mapped during this step (0 when the map shrank, e.g. after a SLAM reset)',
}


def read_session(path: str) -> dict:
    """All the messages the steps are built from, as time-sorted arrays (times in ns)."""
    series = {name: ([], []) for name in (
        'scan', 'cmd', 'joy', 'odom', 'wheel_vels', 'wheel_status', 'mouse', 'slip', 'ir', 'kidnap', 'dock',
        'battery', 'rescue', 'map_odom', 'odom_base', 'map', 'hazard')}
    meta = {}
    maps = []  # (time, grid, origin, resolution) for each SLAM map
    raw_scans = []  # (time, ranges, angle_min, angle_increment, range_max, frame)
    static = {}  # frame -> (parent, yaw), from /tf_static
    rescue_held = False

    def add(name, t, value):
        series[name][0].append(t)
        series[name][1].append(value)

    for m in read_ros2_messages(path):
        t, topic, msg = m.log_time_ns, m.channel.topic, m.ros_msg
        if topic == '/scan':
            # Kept raw until the lidar's mounting is known (from /tf_static, which can come later).
            raw_scans.append((t, np.asarray(msg.ranges, dtype=np.float32), msg.angle_min, msg.angle_increment,
                              msg.range_max, msg.header.frame_id))
            meta.setdefault('scan_range_max', msg.range_max)
        elif topic == '/tf_static':
            for tf in msg.transforms:
                r = tf.transform.rotation
                static[tf.child_frame_id] = (tf.header.frame_id, yaw_from_quaternion(r.x, r.y, r.z, r.w))
        elif topic == '/cmd_vel':
            add('cmd', t, (msg.twist.linear.x, msg.twist.angular.z))
        elif topic == '/joy':
            # Driving with the controller means holding its enable button; the triggers rest at
            # +-1, so stick positions alone don't tell.
            add('joy', t, any(msg.buttons))
            pressed = len(msg.buttons) > RESCUE_BUTTON and msg.buttons[RESCUE_BUTTON] == 1
            if pressed and not rescue_held:
                add('rescue', t, True)
            rescue_held = pressed
        elif topic == '/odom':
            add('odom', t, (msg.twist.twist.linear.x, msg.twist.twist.angular.z))
        elif topic == '/wheel_vels':
            add('wheel_vels', t, (msg.velocity_left, msg.velocity_right))
        elif topic == '/wheel_status':
            add('wheel_status', t, (msg.current_ma_left, msg.current_ma_right))
        elif topic == '/mouse':
            add('mouse', t, (msg.integrated_x, msg.integrated_y, msg.last_squal))
        elif topic == '/ir_intensity':
            values = np.full(len(IR_FRAMES), np.nan, dtype=np.float32)
            for reading in msg.readings:
                if reading.header.frame_id in IR_FRAMES:
                    values[IR_FRAMES.index(reading.header.frame_id)] = reading.value
            add('ir', t, values)
        elif topic == '/slip_status':
            add('slip', t, msg.is_slipping)
        elif topic == '/kidnap_status':
            add('kidnap', t, msg.is_kidnapped)
        elif topic == '/dock_status':
            add('dock', t, msg.is_docked)
        elif topic == '/battery_state':
            add('battery', t, msg.percentage)
        elif topic == '/hazard_detection':
            for d in msg.detections:
                add('hazard', t, (d.type, d.header.frame_id))
        elif topic == '/tf':
            for tf in msg.transforms:
                pose = (tf.transform.translation.x, tf.transform.translation.y,
                        yaw_from_quaternion(tf.transform.rotation.x, tf.transform.rotation.y,
                                            tf.transform.rotation.z, tf.transform.rotation.w))
                if (tf.header.frame_id, tf.child_frame_id) == ('map', 'odom'):
                    add('map_odom', t, pose)
                elif (tf.header.frame_id, tf.child_frame_id) == ('odom', 'base_link'):
                    add('odom_base', t, pose)
        elif topic == '/map':
            raw = msg.data
            data = np.frombuffer(raw, dtype=np.int8) if isinstance(raw, (bytes, bytearray)) else np.asarray(raw, dtype=np.int8)
            add('map', t, int((data >= 0).sum()))
            meta['map_resolution'] = msg.info.resolution
            maps.append((t, data.reshape(msg.info.height, msg.info.width).copy(),
                         (msg.info.origin.position.x, msg.info.origin.position.y), msg.info.resolution))

    def yaw_in_base(frame):
        yaw = 0.0
        while frame != 'base_link':
            if frame not in static:
                raise ValueError(f'no /tf_static path from {frame} to base_link')
            frame, step = static[frame]
            yaw += step
        return yaw

    for t, ranges, angle_min, inc, range_max, frame in raw_scans:
        add('scan', t, robot_sectors(ranges, angle_min, inc, yaw_in_base(frame), range_max, SCAN_SECTORS))
    if raw_scans:
        meta['lidar_mount_yaw'] = yaw_in_base(raw_scans[0][5])

    # The bag's messages arrive roughly but not strictly in time order.
    out = {}
    for name, (times, values) in series.items():
        order = np.argsort(np.asarray(times, dtype=np.int64), kind='stable')
        out[name] = (np.asarray(times, dtype=np.int64)[order], [values[i] for i in order])
    out['meta'] = meta
    out['maps'] = sorted(maps, key=lambda m: m[0])
    return out


def save_maps(path: str, maps, start_ns: int):
    """Every SLAM map of a session in one file: times (s since the first scan), origins,
    resolutions, and the grids flattened end to end (load them with pet_learn.maps.load_maps)."""
    grids = [g for _, g, _, _ in maps]
    np.savez_compressed(
        path,
        t=np.array([(t - start_ns) / 1e9 for t, _, _, _ in maps]),
        origin=np.array([o for _, _, o, _ in maps], dtype=np.float64).reshape(-1, 2),
        resolution=np.array([r for _, _, _, r in maps], dtype=np.float64),
        shape=np.array([g.shape for g in grids], dtype=np.int64).reshape(-1, 2),
        data=np.concatenate([g.ravel() for g in grids]) if grids else np.zeros(0, dtype=np.int8),
    )


def build_steps(s: dict) -> dict:
    scan_times = s['scan'][0]
    if len(scan_times) == 0:
        raise ValueError('no lidar scans in this session')
    start, end = scan_times[0], scan_times[-1]
    step_ns = int(1e9 / RATE)
    grid = np.arange(start, end + 1, step_ns, dtype=np.int64)
    n = len(grid)

    def latest(name, default, max_age_s=None):
        """The latest value of a series at each step (default if none yet, or older than max_age_s)."""
        times, values = s[name]
        idx = latest_before(times, grid)
        result = []
        for i, k in enumerate(idx):
            fresh = k >= 0 and (max_age_s is None or grid[i] - times[k] <= max_age_s * 1e9)
            result.append(values[k] if fresh else default)
        return np.array(result)

    def during_step(name, match):
        """Whether a message matching match() arrived during each step (t - step, t]."""
        times, values = s[name]
        hit = np.zeros(n, dtype=bool)
        for t, v in zip(times, values):
            i = int(math.ceil((t - start) / step_ns))
            if 0 <= i < n and match(v):
                hit[i] = True
        return hit

    steps = {
        't': (grid - start) / 1e9,
        'scan': latest('scan', np.full(SCAN_SECTORS, np.nan, dtype=np.float32)).astype(np.float32),
        'cmd': latest('cmd', (0.0, 0.0), COMMAND_TIMEOUT).astype(np.float32),
        'joy_active': latest('joy', False, COMMAND_TIMEOUT).astype(bool),
        'odom_vel': latest('odom', (0.0, 0.0)).astype(np.float32),
        'wheel_vels': latest('wheel_vels', (0.0, 0.0)).astype(np.float32),
        'wheel_current': latest('wheel_status', (0, 0)).astype(np.float32),
        'slipping': latest('slip', False).astype(bool),
        'ir': latest('ir', np.full(len(IR_FRAMES), np.nan, dtype=np.float32)).astype(np.float32),
        'kidnapped': latest('kidnap', False).astype(bool),
        'docked': latest('dock', False).astype(bool),
        'battery': latest('battery', np.nan).astype(np.float32),
        'rescue': during_step('rescue', lambda v: v),
        'stall': during_step('hazard', lambda h: h[0] == HAZARD_TYPES['stall']),
        'cliff_or_drop': during_step('hazard', lambda h: h[0] in (HAZARD_TYPES['cliff'], HAZARD_TYPES['wheel_drop'])),
    }

    # Bumps: the base reports one only when it starts (often twice at once), so mark the step it began.
    bump = np.zeros((n, 3), dtype=bool)
    for side in range(3):
        bump[:, side] = during_step('hazard', lambda h, side=side: h[0] == HAZARD_TYPES['bump'] and bump_side(h[1]) == side)
    steps['bump'] = bump

    # Floor sensor: it reports total distance moved; speed is the change over one step.
    mouse = latest('mouse', (np.nan, np.nan, 0)).astype(np.float64)
    moved = np.hypot(np.diff(mouse[:, 0], prepend=mouse[0, 0]), np.diff(mouse[:, 1], prepend=mouse[0, 1]))
    steps['mouse_speed'] = (moved * RATE).astype(np.float32)
    steps['mouse_quality'] = mouse[:, 2].astype(np.float32)

    # SLAM pose: SLAM's map -> odom correction applied to the wheels' odom -> base_link.
    none = (np.nan, np.nan, np.nan)
    map_odom = latest('map_odom', none)
    odom_base = latest('odom_base', none)
    pose = np.full((n, 3), np.nan, dtype=np.float32)
    for i in range(n):
        if not (np.isnan(map_odom[i]).any() or np.isnan(odom_base[i]).any()):
            pose[i] = compose(map_odom[i], odom_base[i])
    steps['pose'] = pose

    known = latest('map', 0).astype(np.int64)
    steps['known_cells'] = known
    steps['new_cells'] = np.maximum(np.diff(known, prepend=known[0]), 0)
    return steps


def summarize(name: str, steps: dict, meta: dict) -> str:
    t = steps['t']
    dt = 1 / RATE
    pose = steps['pose']
    jumps = np.hypot(*np.diff(pose[:, :2], axis=0).T)
    driven = float(np.nansum(np.where(jumps < 0.5, jumps, 0)))  # bigger jumps are SLAM corrections or resets
    forward_cmd = steps['cmd'][:, 0] > 0.03
    pushing = forward_cmd & (steps['mouse_speed'] < 0.01) & ~steps['docked']
    battery = steps['battery'][~np.isnan(steps['battery'])]
    area = steps['known_cells'].max() * meta.get('map_resolution', 0.05) ** 2
    return '\n'.join([
        f'{name}: {t[-1] / 60:.1f} min, {len(t)} steps',
        f'  driven {driven:.1f} m (SLAM), a person driving {steps["joy_active"].mean() * 100:.0f}% of the time',
        f'  bumps {int(steps["bump"].any(axis=1).sum())} (left {int(steps["bump"][:, 0].sum())}, '
        f'centre {int(steps["bump"][:, 1].sum())}, right {int(steps["bump"][:, 2].sum())}), '
        f'stall reports in {int(steps["stall"].sum())} steps, rescues {int(steps["rescue"].sum())}',
        f'  told to go forward but not moving over the floor: {pushing.sum() * dt:.0f} s; '
        f'wheel slip reported: {steps["slipping"].sum() * dt:.0f} s',
        f'  mapped {area:.1f} m² at most; battery {battery[0] * 100:.0f}% → {battery[-1] * 100:.0f}%',
    ])


def convert(session_dir: str, out_path: str) -> str:
    bags = sorted(glob.glob(os.path.join(session_dir, '*.mcap')))
    if not bags:
        raise ValueError(f'no .mcap file in {session_dir}')
    s = read_session(bags[0])
    steps = build_steps(s)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta = s['meta']
    np.savez_compressed(out_path, rate=RATE, scan_range_max=meta.get('scan_range_max', np.nan),
                        lidar_mount_yaw=meta.get('lidar_mount_yaw', np.nan), **steps)
    save_maps(out_path.replace('.npz', '.maps.npz'), s['maps'], s['scan'][0][0])
    return summarize(os.path.basename(session_dir.rstrip('/')), steps, meta)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.normpath(os.path.join(here, '..', '..', 'data'))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--drives', default=os.path.join(data, 'drives'))
    parser.add_argument('--out', default=os.path.join(data, 'steps'))
    parser.add_argument('--force', action='store_true', help='convert sessions that already have steps too')
    args = parser.parse_args()

    sessions = sorted(d for d in glob.glob(os.path.join(args.drives, '*')) if os.path.isdir(d))
    if not sessions:
        print(f'no sessions in {args.drives}; record one and run ops/pull-drives.sh')
    for session in sessions:
        out = os.path.join(args.out, os.path.basename(session) + '.npz')
        if os.path.exists(out) and not args.force:
            print(f'{os.path.basename(session)}: already converted')
            continue
        print(convert(session, out))
        print(f'  → {out}')


if __name__ == '__main__':
    main()
