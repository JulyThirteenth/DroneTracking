"""Launch the CCM reference and controller nodes."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Build the common SITL and real-vehicle launch description."""
    package_share = Path(get_package_share_directory("drone_ccm"))
    parameters = str(package_share / "config" / "ccm.yaml")
    checkpoint = LaunchConfiguration("checkpoint")
    reference_mode = LaunchConfiguration("reference_mode")
    waypoint_file = LaunchConfiguration("waypoint_file")
    waypoint_speed = LaunchConfiguration("waypoint_speed")
    device = LaunchConfiguration("device")
    hover_thrust = LaunchConfiguration("hover_thrust")
    return LaunchDescription(
        (
            DeclareLaunchArgument(
                "checkpoint",
                default_value=str(
                    package_share / "models" / "neu_ccm_practical.pt"
                ),
                description="Installed model basename or checkpoint file path.",
            ),
            DeclareLaunchArgument(
                "reference_mode",
                default_value="hover",
                description="Reference generator: hover or waypoint.",
            ),
            DeclareLaunchArgument(
                "waypoint_file",
                default_value="",
                description="NED waypoint file required by waypoint mode.",
            ),
            DeclareLaunchArgument(
                "waypoint_speed",
                default_value="1.0",
                description="Maximum three-dimensional waypoint speed in m/s.",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cpu",
                description="PyTorch inference device.",
            ),
            DeclareLaunchArgument(
                "hover_thrust",
                default_value="0.5812",
                description="Normalized PX4 thrust that balances vehicle weight.",
            ),
            Node(
                package="drone_ccm",
                executable="drone_ccm_reference",
                name="drone_ccm_reference",
                output="screen",
                parameters=[
                    parameters,
                    {
                        "checkpoint": checkpoint,
                        "mode": reference_mode,
                        "waypoint_file": waypoint_file,
                        "waypoint_speed": waypoint_speed,
                    },
                ],
            ),
            Node(
                package="drone_ccm",
                executable="drone_ccm_controller",
                name="drone_ccm_controller",
                output="screen",
                parameters=[
                    parameters,
                    {
                        "checkpoint": checkpoint,
                        "device": device,
                        "hover_thrust": hover_thrust,
                    },
                ],
            ),
        )
    )
