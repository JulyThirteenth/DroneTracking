#!/usr/bin/env bash
source /opt/ros/foxy/setup.bash && source ~/wsa/ws_px4/install/setup.bash
ros2 topic echo /fmu/out/vehicle_local_position