"""Launch the contour-following MPCC controller."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Build the MPCC controller launch description."""
    default_config = os.path.join(
        get_package_share_directory("drone_cnt"),
        "config",
        "mpcc.yaml",
    )
    config_file = LaunchConfiguration("config_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Absolute path to the MPCC parameter file.",
            ),
            Node(
                package="drone_cnt",
                executable="drone_cnt_mpcc",
                name="drone_cnt_mpcc",
                output="screen",
                parameters=[config_file],
            ),
        ]
    )
