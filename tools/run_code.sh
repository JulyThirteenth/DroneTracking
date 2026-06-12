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

# 已存在会话则直接进入
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  if [[ -n "${CONFIG_FILE}" ]]; then
    echo "[INFO] tmux session '${SESSION}' already exists. Existing processes keep old config."
    echo "[INFO] Recreate session to apply DRONE_TRACKING_CONFIG=${CONFIG_FILE}."
  fi
  tmux attach -t "${SESSION}"
  exit 0
fi

# 新建会话并竖向分成三个 pane（上下堆叠）
tmux new-session -d -s "${SESSION}" -n control "bash"
tmux split-window -v -t "${SESSION}":0.0
tmux split-window -v -t "${SESSION}":0.1
tmux select-layout -t "${SESSION}":0 even-vertical

ENV_PREFIX=""
if [[ -n "${CONFIG_FILE}" ]]; then
  ENV_PREFIX="export DRONE_TRACKING_CONFIG=\"${CONFIG_FILE}\" && "
  echo "[INFO] DRONE_TRACKING_CONFIG=${CONFIG_FILE}"
fi

tmux send-keys -t "${SESSION}":0.0 \
"cd \"${PROJECT_DIR}\" && source ./tools/simdrone_env.sh && ${ENV_PREFIX}python plan2track/plan2track.py" C-m

tmux send-keys -t "${SESSION}":0.1 \
"cd \"${PROJECT_DIR}\" && source ./tools/simdrone_env.sh && ${ENV_PREFIX}python -m fsm.fsm_main" C-m

tmux send-keys -t "${SESSION}":0.2 \
"cd \"${PROJECT_DIR}\" && source ./tools/simdrone_env.sh && ${ENV_PREFIX}python -m fsm.fsm_interface" C-m

tmux select-pane -t "${SESSION}":0.2
tmux attach -t "${SESSION}"
