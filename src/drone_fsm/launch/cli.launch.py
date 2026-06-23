"""Launch the interactive drone FSM command terminal."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Build the drone CLI launch description."""
    default_config = os.path.join(
        get_package_share_directory("drone_fsm"),
        "config",
        "drone_fsm.yaml",
    )
    terminal = None
    terminal_environment = {}
    try:
        terminal = os.ttyname(0)
        terminal_environment = {
            "DRONE_CLI_TTY": terminal,
            "DRONE_CLI_PROMPT_PREFIX": "[drone_cli-1] ",
        }
    except OSError:
        pass

    config_file = LaunchConfiguration("config_file")
    cli_node = Node(
        package="drone_fsm",
        executable="drone_cli",
        name="drone_cli",
        output="screen",
        emulate_tty=True,
        additional_env=terminal_environment,
        parameters=[config_file],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Absolute path to the drone_fsm parameter file.",
            ),
            cli_node,
        ]
    )
