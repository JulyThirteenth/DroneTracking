#!/usr/bin/env bash
# 用法：source ./tools/simulate_env.sh  或  . ./tools/simulate_env.sh

# 如果你用 bash 执行而不是 source，就提醒
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: 请用 source 运行："
  echo "  source ./tools/simulate_env.sh"
  return 1 2>/dev/null || exit 1
fi

# 1) 让 conda activate 在当前 shell 可用
# 按你的实际 conda 安装路径二选一（miniconda3/anaconda3）
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: 找不到 conda.sh，请确认 conda 安装路径（miniconda3/anaconda3）"
  return 1
fi

# 2) 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate mpcc

# 3) source ROS2 和工作空间
source /opt/ros/humble/setup.bash
source ~/devspace/FlightSim/px4-ros_ws/install/setup.bash

# 4) 打印确认信息
echo "[OK] CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "[OK] ROS_DISTRO=$ROS_DISTRO"
echo "[OK] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<unset>}"

