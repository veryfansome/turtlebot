"""Start everything the explorer needs: SLAM, Nav2 with this repo's config/nav2.yaml, and the explorer.

    ros2 launch pet_explorer explore.launch.py [rviz:=true] [nav2_params:=...] [slam_params:=...]

SLAM and Nav2 are started through turtlebot4_navigation's own launch files, so the TurtleBot 4
defaults apply except where our config files override them.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    tb4_nav = get_package_share_directory('turtlebot4_navigation')
    tb4_viz = get_package_share_directory('turtlebot4_viz')
    # The repo's config/nav2.yaml. The repo root is the colcon workspace, and this package installs to
    # <workspace>/install/pet_explorer/share/pet_explorer.
    workspace = os.path.join(get_package_share_directory('pet_explorer'), '..', '..', '..', '..')
    default_nav2_params = os.path.normpath(os.path.join(workspace, 'config', 'nav2.yaml'))

    return LaunchDescription([
        DeclareLaunchArgument('nav2_params', default_value=default_nav2_params,
                              description='Nav2 parameters file'),
        DeclareLaunchArgument('slam_params', default_value=os.path.join(tb4_nav, 'config', 'slam.yaml'),
                              description='slam_toolbox parameters file'),
        DeclareLaunchArgument('rviz', default_value='false', choices=['true', 'false'],
                              description='Also open RViz with the navigation view'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb4_nav, 'launch', 'slam.launch.py')),
            launch_arguments={'params': LaunchConfiguration('slam_params')}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb4_nav, 'launch', 'nav2.launch.py')),
            launch_arguments={'params_file': LaunchConfiguration('nav2_params')}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb4_viz, 'launch', 'view_navigation.launch.py')),
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
        Node(package='pet_explorer', executable='explorer', output='screen'),
    ])
