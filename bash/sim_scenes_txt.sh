#!/usr/bin/env bash
set -euo pipefail

SESSION="${SIMDRONE_SESSION:-dronesim}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux attach -t "$SESSION"
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_SH="$SCRIPT_DIR/simdrone_env.sh"
QGC_PATH="${QGC_PATH:-$HOME/DroneSimulator/QGroundControl-x86_64.AppImage}"

SIM_ARGS=""
if (($# > 0)); then
  printf -v SIM_ARGS ' %q' "$@"
fi

tmux new-session -d -s "$SESSION" -n main "bash"

P_TOP="$(tmux display-message -p -t "$SESSION":0.0 "#{pane_id}")"
P_BOTTOM="$(tmux split-window -v -p 90 -t "$P_TOP" -P -F "#{pane_id}")"
P_RVIZ="$P_TOP"
P_TF_TREE="$(tmux split-window -h -p 50 -t "$P_RVIZ" -P -F "#{pane_id}")"

P_ISAAC="$P_BOTTOM"
P_AGENT="$(tmux split-window -h -t "$P_ISAAC" -P -F "#{pane_id}")"
P_QGC="$(tmux split-window -h -t "$P_AGENT" -P -F "#{pane_id}")"

tmux send-keys -t "$P_RVIZ" \
"cd \"$PROJECT_DIR\" && source \"$ENV_SH\" && rviz2 -d \"$PROJECT_DIR/layout.rviz\"" C-m

tmux send-keys -t "$P_TF_TREE" \
"cd \"$PROJECT_DIR\" && source \"$ENV_SH\" && python isaacsim/sim_tf_tree.py" C-m

tmux send-keys -t "$P_ISAAC" \
"cd \"$PROJECT_DIR\" && isaac_run isaacsim/sim_scenes_txt.py$SIM_ARGS" C-m

tmux send-keys -t "$P_AGENT" \
"MicroXRCEAgent udp4 -p 8888" C-m

tmux send-keys -t "$P_QGC" \
"\"$QGC_PATH\"" C-m

tmux select-pane -t "$P_ISAAC"
tmux attach -t "$SESSION"
