#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate mpcc
source /opt/ros/humble/setup.bash
source ~/IsaacSim-ros_workspaces/humble_ws/install/setup.bash
exec "$CONDA_PREFIX/bin/python" -m ipykernel_launcher "$@"