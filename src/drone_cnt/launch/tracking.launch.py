"""Launch a complete MPC or MPCC tracking and logging stack."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def _source(package: str, launch_file: str) -> PythonLaunchDescriptionSource:
    path = Path(get_package_share_directory(package)) / "launch" / launch_file
    return PythonLaunchDescriptionSource(str(path))


def _mode_is(mode: LaunchConfiguration, expected: str) -> IfCondition:
    return IfCondition(PythonExpression(["'", mode, "' == '", expected, "'"]))


def generate_launch_description() -> LaunchDescription:
    """Build the selected controller, reference generator and logger stack."""
    cnt_share = get_package_share_directory("drone_cnt")
    cnt_mpc_config = os.path.join(cnt_share, "config", "mpc.yaml")
    cnt_mpcc_config = os.path.join(cnt_share, "config", "mpcc.yaml")

    mode = LaunchConfiguration("controller_mode")
    path_file = LaunchConfiguration("path_file")
    reference_speed = LaunchConfiguration("reference_speed")
    log_directory = LaunchConfiguration("log_directory")
    run_name = LaunchConfiguration("run_name")
    return LaunchDescription(
        (
            DeclareLaunchArgument(
                "controller_mode",
                default_value="mpc",
                description="Tracking controller: mpc or mpcc.",
            ),
            DeclareLaunchArgument(
                "path_file",
                default_value="waypoints/line_waypoint.txt",
                description=(
                    "Absolute path or drone_ref share-relative waypoint file."
                ),
            ),
            DeclareLaunchArgument(
                "reference_speed",
                default_value="3.0",
                description="MPC reference speed in m/s; unused by MPCC.",
            ),
            DeclareLaunchArgument(
                "log_directory",
                default_value="logs",
                description="Directory containing flight-log run directories.",
            ),
            DeclareLaunchArgument(
                "run_name",
                default_value="",
                description="Optional unique flight-log run name.",
            ),
            IncludeLaunchDescription(
                _source("drone_ref", "mpc.launch.py"),
                condition=_mode_is(mode, "mpc"),
                launch_arguments={
                    "path_file": path_file,
                    "reference_speed": reference_speed,
                }.items(),
            ),
            IncludeLaunchDescription(
                _source("drone_cnt", "mpc.launch.py"),
                condition=_mode_is(mode, "mpc"),
                launch_arguments={"config_file": cnt_mpc_config}.items(),
            ),
            IncludeLaunchDescription(
                _source("drone_ref", "mpcc.launch.py"),
                condition=_mode_is(mode, "mpcc"),
                launch_arguments={"path_file": path_file}.items(),
            ),
            IncludeLaunchDescription(
                _source("drone_cnt", "mpcc.launch.py"),
                condition=_mode_is(mode, "mpcc"),
                launch_arguments={"config_file": cnt_mpcc_config}.items(),
            ),
            IncludeLaunchDescription(
                _source("drone_log", "log.launch.py"),
                launch_arguments={
                    "controller_mode": mode,
                    "log_directory": log_directory,
                    "run_name": run_name,
                }.items(),
            ),
        )
    )
