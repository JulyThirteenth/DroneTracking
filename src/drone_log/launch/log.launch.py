"""Launch the independent MPC or MPCC flight logger."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Build the flight-log launch description."""
    config = (
        Path(get_package_share_directory("drone_log")) / "config" / "drone_log.yaml"
    )
    return LaunchDescription(
        (
            DeclareLaunchArgument("controller_mode", default_value="mpc"),
            DeclareLaunchArgument("log_directory", default_value="logs"),
            DeclareLaunchArgument("run_name", default_value=""),
            DeclareLaunchArgument("flush_interval", default_value="20"),
            Node(
                package="drone_log",
                executable="drone_log",
                name="drone_log",
                output="screen",
                parameters=[
                    str(config),
                    {
                        "controller_mode": LaunchConfiguration("controller_mode"),
                        "log_directory": LaunchConfiguration("log_directory"),
                        "run_name": LaunchConfiguration("run_name"),
                        "flush_interval": LaunchConfiguration("flush_interval"),
                    },
                ],
            ),
        )
    )
