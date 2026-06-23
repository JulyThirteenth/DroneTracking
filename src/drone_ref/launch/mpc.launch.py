"""Launch the MPC reference trajectory publisher."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """Build the MPC reference launch description."""
    default_config = os.path.join(
        get_package_share_directory("drone_ref"),
        "config",
        "drone_ref.yaml",
    )
    config_file = LaunchConfiguration("config_file")
    path_file = LaunchConfiguration("path_file")
    reference_speed = LaunchConfiguration("reference_speed")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Absolute path to the MPC parameter file.",
            ),
            DeclareLaunchArgument(
                "path_file",
                default_value="waypoints/line_waypoint.txt",
                description="Absolute path or drone_ref share-relative path.",
            ),
            DeclareLaunchArgument(
                "reference_speed",
                default_value="3.0",
                description="Reference speed in metres per second.",
            ),
            Node(
                package="drone_ref",
                executable="drone_ref_mpc",
                name="drone_ref_mpc",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "path_file": path_file,
                        "reference_speed": ParameterValue(
                            reference_speed,
                            value_type=float,
                        ),
                    },
                ],
            ),
        ]
    )
