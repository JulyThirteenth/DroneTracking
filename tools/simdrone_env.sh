#!/usr/bin/env bash
# Usage:
#   source ./tools/simdrone_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: please source this file:"
  echo "  source ./tools/simdrone_env.sh"
  exit 1
fi

_SIMDRONE_ENV_CWD="$(pwd)"

if [[ -f /root/gpufree-data/workspace/activate_drone_rl.sh ]]; then
  source /root/gpufree-data/workspace/activate_drone_rl.sh
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate /root/gpufree-data/conda-envs/drone-rl
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate mpcc
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate mpcc
else
  echo "ERROR: cannot find conda activation script"
  return 1
fi

cd "${_SIMDRONE_ENV_CWD}"
unset _SIMDRONE_ENV_CWD

if [[ -f /root/gpufree-data/devspace/drone/setup_drone_stack.sh ]]; then
  source /root/gpufree-data/devspace/drone/setup_drone_stack.sh
fi

source /opt/ros/humble/setup.bash

if [[ -f /root/gpufree-data/devspace/drone/IsaacSim-ros_workspaces/humble_ws/install/setup.bash ]]; then
  source /root/gpufree-data/devspace/drone/IsaacSim-ros_workspaces/humble_ws/install/setup.bash
elif [[ -f "$HOME/devspace/FlightSim/px4-ros_ws/install/setup.bash" ]]; then
  source "$HOME/devspace/FlightSim/px4-ros_ws/install/setup.bash"
fi

if [[ -n "${ISAACSIM_PYTHON:-}" ]]; then
  isaac_run() {
    "${ISAACSIM_PYTHON}" "$@"
  }
  export -f isaac_run
fi

echo "[OK] python=$(command -v python)"
echo "[OK] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
echo "[OK] ROS_DISTRO=${ROS_DISTRO:-<unset>}"
echo "[OK] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<unset>}"
echo "[OK] ISAACSIM_PATH=${ISAACSIM_PATH:-<unset>}"

