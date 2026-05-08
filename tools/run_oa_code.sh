#!/usr/bin/env bash

set -euo pipefail

SESSION="${1:-dronecnt}"
CONFIG_FILE="${2:-${DRONE_TRACKING_CONFIG:-}}"
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
  ENV_PREFIX="export DRONE_TRACKING_CONFIG=\"${CONFIG_FILE}\" && "
  echo "[INFO] DRONE_TRACKING_CONFIG=${CONFIG_FILE}"
fi

DEPTH_TOPIC="${DEPTH_TOPIC:-/depth}"
DEPTH_SCAN_TOPIC="${DEPTH_SCAN_TOPIC:-/depth2scan/scan}"
DEPTH_POINTS_TOPIC="${DEPTH_POINTS_TOPIC:-/depth2scan/points}"
DEPTH_FRAME_ID="${DEPTH_FRAME_ID:-drone_fpv_camera}"
DEPTH_SCAN_CONFIG="${DEPTH_SCAN_CONFIG:-${PROJECT_DIR}/perception/yaml/depth_transform.yaml}"

tmux send-keys -t "${P_PLAN}" \
"cd \"${PROJECT_DIR}\" && source ./tools/simdrone_env.sh && ${ENV_PREFIX}python plan2track/keyboard2track.py" C-m

tmux send-keys -t "${P_FSM_NODE}" \
"cd \"${PROJECT_DIR}\" && source ./tools/simdrone_env.sh && ${ENV_PREFIX}python fsm/fsm_node.py" C-m

tmux send-keys -t "${P_FSM_APP}" \
"cd \"${PROJECT_DIR}\" && source ./tools/simdrone_env.sh && ${ENV_PREFIX}python fsm/fsm_app.py" C-m

tmux send-keys -t "${P_BOTTOM}" \
"cd \"${PROJECT_DIR}\" && source ./tools/simdrone_env.sh && python perception/depth2scan.py --ros-args -p depth_topic:=\"${DEPTH_TOPIC}\" -p scan_topic:=\"${DEPTH_SCAN_TOPIC}\" -p points_topic:=\"${DEPTH_POINTS_TOPIC}\" -p frame_id:=\"${DEPTH_FRAME_ID}\" -p config_path:=\"${DEPTH_SCAN_CONFIG}\"" C-m

tmux select-pane -t "${P_FSM_APP}"
tmux attach -t "${SESSION}"
