"""Launch the FSM state owner and PX4 flight executive."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Build the drone FSM launch description."""
    default_config = os.path.join(
        get_package_share_directory("drone_fsm"),
        "config",
        "drone_fsm.yaml",
    )
    config_file = LaunchConfiguration("config_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Absolute path to the drone_fsm parameter file.",
            ),
            Node(
                package="drone_fsm",
                executable="drone_fsm",
                name="drone_fsm",
                output="screen",
                parameters=[config_file],
            ),
            Node(
                package="drone_fsm",
                executable="drone_fly",
                name="drone_fly",
                output="screen",
                parameters=[config_file],
            ),
        ]
    )
