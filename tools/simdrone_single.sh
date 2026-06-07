#!/usr/bin/env bash

SESSION="dronesim"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux attach -t "$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n main "bash"

# Top row: RViz and TF tree. Bottom row: Isaac, MicroXRCEAgent, and QGC.
P_TOP="$(tmux display-message -p -t "$SESSION":0.0 "#{pane_id}")"
P_BOTTOM="$(tmux split-window -v -p 90 -t "$P_TOP" -P -F "#{pane_id}")"
P_RVIZ="$P_TOP"
P_TF_TREE="$(tmux split-window -h -p 50 -t "$P_RVIZ" -P -F "#{pane_id}")"

P_ISAAC="$P_BOTTOM"
P_AGENT="$(tmux split-window -h -t "$P_ISAAC" -P -F "#{pane_id}")"
P_QGC="$(tmux split-window -h -t "$P_AGENT" -P -F "#{pane_id}")"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_SH="$SCRIPT_DIR/simdrone_env.sh"
MTL_SH="$SCRIPT_DIR/export_mtl.sh"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

QGC_APP="${QGC_APP:-}"
if [[ -z "$QGC_APP" ]]; then
  for candidate in \
    "$HOME/DroneSimulator/QGroundControl-x86_64.AppImage" \
    "/root/gpufree-data/devspace/drone/QGroundControl-x86_64.AppImage" \
    "/root/gpufree-data/QGroundControl-x86_64.AppImage"; do
    if [[ -x "$candidate" ]]; then
      QGC_APP="$candidate"
      break
    fi
  done
fi

SIM_ARGS=()
if [[ -n "${SIM_SCENE_SOURCE:-}" ]]; then
  SIM_ARGS+=(--scene-source "${SIM_SCENE_SOURCE}")
fi
if [[ -n "${SIM_SCENE_INDEX:-}" ]]; then
  SIM_ARGS+=(--scene-index "${SIM_SCENE_INDEX}")
fi
if [[ -n "${SIM_TASK_INDEX:-}" ]]; then
  SIM_ARGS+=(--task-index "${SIM_TASK_INDEX}")
fi
if [[ -n "${SIM_SPAWN_CLEARANCE:-}" ]]; then
  SIM_ARGS+=(--spawn-clearance "${SIM_SPAWN_CLEARANCE}")
fi
if [[ -n "${SIM_SPAWN_SEED:-}" ]]; then
  SIM_ARGS+=(--spawn-seed "${SIM_SPAWN_SEED}")
fi
printf -v SIM_ARGS_Q '%q ' "${SIM_ARGS[@]}"

tmux send-keys -t "$P_RVIZ" \
"cd \"$PROJECT_DIR\" && source \"$ENV_SH\" && rviz2 -d \"$SCRIPT_DIR/layout.rviz\"" C-m

tmux send-keys -t "$P_TF_TREE" \
"cd \"$PROJECT_DIR\" && source \"$ENV_SH\" && python isaacsim/tf_tree.py" C-m

tmux send-keys -t "$P_ISAAC" \
"cd \"$PROJECT_DIR\" && source \"$ENV_SH\" && source \"$MTL_SH\" && isaac_run isaacsim/sim_single.py ${SIM_ARGS_Q}" C-m

tmux send-keys -t "$P_AGENT" \
"cd \"$PROJECT_DIR\" && source \"$ENV_SH\" && MicroXRCEAgent udp4 -p 8888" C-m

if [[ -n "$QGC_APP" ]]; then
  tmux send-keys -t "$P_QGC" \
  "\"$QGC_APP\"" C-m
else
  tmux send-keys -t "$P_QGC" \
  "echo '[WARN] QGroundControl AppImage not found; skipping QGC pane.'" C-m
fi

tmux select-pane -t "$P_ISAAC"
tmux attach -t "$SESSION"
