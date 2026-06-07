#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SESSION="${SESSION:-dronecnt}"
CONFIG_FILE="${DRONE_RACING_CONFIG:-${DRONE_TRACKING_CONFIG:-yamls/config_rl_goal.yaml}}"
SETTLE_S="${SETTLE_S:-8}"
TAKEOFF_S="${TAKEOFF_S:-8}"
EXECUTE_S="${EXECUTE_S:-25}"
FIXED_GOAL_OFFSET_ENU="${FIXED_GOAL_OFFSET_ENU:-1.0,0.0,0.0}"
FIXED_GOAL_Z="${FIXED_GOAL_Z:-1.0}"
FIXED_GOAL_ENU="${FIXED_GOAL_ENU:-}"

cd "${PROJECT_DIR}"
set +u
source ./tools/simdrone_env.sh
set -u

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
fi

export DRONE_TRACKING_CONFIG="${CONFIG_FILE}"
export DRONE_RACING_CONFIG="${CONFIG_FILE}"
export PLAN2TRACK_MODE=fixed_goal
export FIXED_GOAL_OFFSET_ENU
export FIXED_GOAL_Z
export FIXED_GOAL_ENU
export ATTACH_CONTROL=0

./tools/run_oa_code.sh "${SESSION}" "${CONFIG_FILE}"

echo "[INFO] waiting ${SETTLE_S}s for ROS nodes"
sleep "${SETTLE_S}"

publish_cmd() {
  local cmd="$1"
  echo "[INFO] fsm cmd: ${cmd}"
  ros2 topic pub --once /fsm/cmd std_msgs/msg/String "{data: ${cmd}}"
}

publish_cmd prepare
sleep 2
publish_cmd takeoff
echo "[INFO] waiting ${TAKEOFF_S}s for takeoff"
sleep "${TAKEOFF_S}"
publish_cmd execute
echo "[INFO] waiting ${EXECUTE_S}s for fixed-goal tracking"
sleep "${EXECUTE_S}"

python3 tools/analyze_rl_goal_log.py --window 300
