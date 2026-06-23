"""Launch the MPCC full-path reference publisher."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Build the MPCC reference launch description."""
    default_config = os.path.join(
        get_package_share_directory("drone_ref"),
        "config",
        "drone_ref.yaml",
    )
    config_file = LaunchConfiguration("config_file")
    path_file = LaunchConfiguration("path_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Absolute path to the MPCC parameter file.",
            ),
            DeclareLaunchArgument(
                "path_file",
                default_value="waypoints/line_waypoint.txt",
                description="Absolute path or drone_ref share-relative path.",
            ),
            Node(
                package="drone_ref",
                executable="drone_ref_mpcc",
                name="drone_ref_mpcc",
                output="screen",
                parameters=[config_file, {"path_file": path_file}],
            ),
        ]
    )
