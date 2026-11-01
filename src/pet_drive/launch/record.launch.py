"""Record a driving session: SLAM, the safety layer, and a bag of everything a driver could learn from.

    ros2 launch pet_drive record.launch.py [out:=~/drives/<time>] [safety:=true] [camera:=false] [slam_params:=...]

Nothing here drives the robot. Drive it with the controller (straight to cmd_vel), or have a driver
publish on drive/cmd_vel through the safety layer; either way the commands end up in the bag.
"""
from datetime import datetime
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

TOPICS = [
    # What the robot senses
    '/scan',  # lidar
    '/odom',  # wheel odometry: what the wheels did
    '/mouse',  # the base's optical floor sensor: what the robot actually did over the floor
    '/imu',  # jolts from collisions the bumper misses
    '/wheel_status',  # motor currents: they rise when it pushes against something
    '/wheel_vels',  # actual wheel speeds
    '/slip_status',  # the base's own wheel-slip detection
    '/ir_intensity',  # the 7 front infrared proximity sensors, at bumper height: things below the lidar
    '/hazard_detection',  # bumps (reported only at the moment of contact), cliffs, wheel drops
    '/kidnap_status',
    '/dock_status',
    '/battery_state',
    # Where it is
    '/tf',  # includes SLAM's map -> odom: movement measured by SLAM, not the wheels
    '/tf_static',
    '/map',
    # What it was told
    '/cmd_vel',  # what the wheels were told, by the safety layer or the controller
    '/drive/cmd_vel',  # what a driver asked the safety layer for
    '/drive/safety',  # why the safety layer changed it
    '/joy',  # the controller: whether a person was driving
]
# Off by default: the robot only offers raw frames, about 5-6 MB/s over Wi-Fi at 30 Hz, enough to
# risk losing lidar scans. Compressed frames would need a camera change on the Pi.
CAMERA_TOPICS = ['/oakd/rgb/preview/image_raw']


def generate_launch_description():
    tb4_nav = get_package_share_directory('turtlebot4_navigation')
    default_out = os.path.expanduser('~/drives/' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    record = ['ros2', 'bag', 'record', '--storage', 'mcap', '--output', LaunchConfiguration('out')]

    return LaunchDescription([
        DeclareLaunchArgument('out', default_value=default_out, description='Bag directory to create'),
        DeclareLaunchArgument('safety', default_value='true', choices=['true', 'false'],
                              description='Also run the safety layer, for drivers using drive/cmd_vel'),
        DeclareLaunchArgument('camera', default_value='false', choices=['true', 'false'],
                              description='Also record the camera preview (raw frames, heavy on Wi-Fi)'),
        DeclareLaunchArgument('slam_params', default_value=os.path.join(tb4_nav, 'config', 'slam.yaml'),
                              description='slam_toolbox parameters file'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb4_nav, 'launch', 'slam.launch.py')),
            launch_arguments={'params': LaunchConfiguration('slam_params')}.items(),
        ),
        Node(package='pet_drive', executable='safety', output='screen',
             condition=IfCondition(LaunchConfiguration('safety'))),
        # Ctrl-C reaches it through ros2 launch and it closes the bag cleanly.
        ExecuteProcess(cmd=record + TOPICS, output='screen', condition=UnlessCondition(LaunchConfiguration('camera'))),
        ExecuteProcess(cmd=record + TOPICS + CAMERA_TOPICS, output='screen',
                       condition=IfCondition(LaunchConfiguration('camera'))),
    ])
