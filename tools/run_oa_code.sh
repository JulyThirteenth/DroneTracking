#!/usr/bin/env bash

set -euo pipefail

SESSION="${1:-dronecnt}"
CONFIG_FILE="${2:-${DRONE_RACING_CONFIG:-${DRONE_TRACKING_CONFIG:-yamls/config_oa.yaml}}}"
ATTACH_CONTROL="${ATTACH_CONTROL:-1}"
if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [tmux_session_name] [config_yaml]"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  if [[ -n "${CONFIG_FILE}" ]]; then
    echo "[INFO] tmux session '${SESSION}' already exists. Existing processes keep old config."
    echo "[INFO] Recreate session to apply DRONE_TRACKING_CONFIG=${CONFIG_FILE}."
  fi
  if [[ "${ATTACH_CONTROL}" == "0" ]]; then
    exit 0
  fi
  tmux attach -t "${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" -n control "bash"
P_PLAN="$(tmux display-message -p -t "${SESSION}":0.0 "#{pane_id}")"
P_BOTTOM="$(tmux split-window -v -p 35 -t "${P_PLAN}" -P -F "#{pane_id}")"
P_FSM_APP="$(tmux split-window -v -p 50 -t "${P_PLAN}" -P -F "#{pane_id}")"
P_FSM_NODE="$(tmux split-window -h -p 50 -t "${P_PLAN}" -P -F "#{pane_id}")"

ENV_PREFIX=""
if [[ -n "${CONFIG_FILE}" ]]; then
  ENV_PREFIX="export DRONE_TRACKING_CONFIG=\"${CONFIG_FILE}\" && export DRONE_RACING_CONFIG=\"${CONFIG_FILE}\" && "
  echo "[INFO] DRONE_RACING_CONFIG=${CONFIG_FILE}"
fi

DEPTH_TOPIC="${DEPTH_TOPIC:-/depth}"
DEPTH_SCAN_TOPIC="${DEPTH_SCAN_TOPIC:-/depth2scan/scan}"
DEPTH_POINTS_TOPIC="${DEPTH_POINTS_TOPIC:-/depth2scan/points}"
DEPTH_FRAME_ID="${DEPTH_FRAME_ID:-drone_fpv_camera}"
DEPTH_SCAN_CONFIG="${DEPTH_SCAN_CONFIG:-${PROJECT_DIR}/perception/yaml/depth_transform.yaml}"
PLAN2TRACK_MODE="${PLAN2TRACK_MODE:-keyboard}"

if [[ "${PLAN2TRACK_MODE}" == "keyboard" ]]; then
  PLAN_CMD="python plan2track/keyboard2track.py"
elif [[ "${PLAN2TRACK_MODE}" == "fixed_goal" ]]; then
  ENV_PREFIX="${ENV_PREFIX}export FIXED_GOAL_OFFSET_ENU=\"${FIXED_GOAL_OFFSET_ENU:-1.0,0.0,0.0}\" && "
  ENV_PREFIX="${ENV_PREFIX}export FIXED_GOAL_Z=\"${FIXED_GOAL_Z:-1.0}\" && "
  ENV_PREFIX="${ENV_PREFIX}export FIXED_GOAL_ENU=\"${FIXED_GOAL_ENU:-}\" && "
  ENV_PREFIX="${ENV_PREFIX}export FIXED_GOAL_PUBLISH_DT=\"${FIXED_GOAL_PUBLISH_DT:-0.05}\" && "
  PLAN_CMD="python plan2track/fixed_goal2track.py"
else
  echo "[ERROR] PLAN2TRACK_MODE must be 'fixed_goal' or 'keyboard', got '${PLAN2TRACK_MODE}'"
  exit 1
fi
echo "[INFO] PLAN2TRACK_MODE=${PLAN2TRACK_MODE}"

tmux send-keys -t "${P_PLAN}" \
"cd \"${PROJECT_DIR}\" && source ./tools/simdrone_env.sh && ${ENV_PREFIX}${PLAN_CMD}" C-m

tmux send-keys -t "${P_FSM_NODE}" \
"cd \"${PROJECT_DIR}\" && source ./tools/simdrone_env.sh && ${ENV_PREFIX}python -m fsm.fsm_main" C-m

tmux send-keys -t "${P_FSM_APP}" \
"cd \"${PROJECT_DIR}\" && source ./tools/simdrone_env.sh && ${ENV_PREFIX}python -m fsm.fsm_interface" C-m

tmux send-keys -t "${P_BOTTOM}" \
"cd \"${PROJECT_DIR}\" && source ./tools/simdrone_env.sh && python perception/depth2scan.py --ros-args -p depth_topic:=\"${DEPTH_TOPIC}\" -p scan_topic:=\"${DEPTH_SCAN_TOPIC}\" -p points_topic:=\"${DEPTH_POINTS_TOPIC}\" -p frame_id:=\"${DEPTH_FRAME_ID}\" -p config_path:=\"${DEPTH_SCAN_CONFIG}\"" C-m

tmux select-pane -t "${P_FSM_APP}"
if [[ "${ATTACH_CONTROL}" != "0" ]]; then
  tmux attach -t "${SESSION}"
fi
