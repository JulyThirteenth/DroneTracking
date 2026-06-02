#!/usr/bin/env bash

SESSION="dronesim"

# 已存在就直接进入
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux attach -t "$SESSION"
  exit 0
fi

# 新建 session
tmux new-session -d -s "$SESSION" -n main "bash"

# 1) 先上下分割：上方占 10%，上方再分两列
P_TOP="$(tmux display-message -p -t "$SESSION":0.0 "#{pane_id}")"
P_BOTTOM="$(tmux split-window -v -p 90 -t "$P_TOP" -P -F "#{pane_id}")"
P_RVIZ="$P_TOP"
P_TF_TREE="$(tmux split-window -h -p 50 -t "$P_RVIZ" -P -F "#{pane_id}")"

# 2) 在下方做三列横向并排
P_ISAAC="$P_BOTTOM"
P_AGENT="$(tmux split-window -h -t "$P_ISAAC" -P -F "#{pane_id}")"
P_QGC="$(tmux split-window -h -t "$P_AGENT" -P -F "#{pane_id}")"

# 3) Pane 0：进入同目录下的 simdrone_env.sh 环境（mpcc + ROS）
# 说明：用 bash -lc 确保 conda 函数可用，然后在里面 source 脚本并进入交互 shell
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_SH="$SCRIPT_DIR/simdrone_env.sh"
MTL_SH="$SCRIPT_DIR/export_mtl.sh"

PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

tmux send-keys -t "$P_RVIZ" \
"cd \"$PROJECT_DIR\" && source \"$ENV_SH\" && rviz2 -d \"$SCRIPT_DIR/layout.rviz\"" C-m

tmux send-keys -t "$P_TF_TREE" \
"cd \"$PROJECT_DIR\" && source \"$ENV_SH\" && python isaacsim/tf_tree.py" C-m


# 4) 下方三个 pane 执行命令
tmux send-keys -t "$P_ISAAC" \
"cd \"$PROJECT_DIR\" && source \"$MTL_SH\" && isaac_run isaacsim/sim_single.py" C-m

tmux send-keys -t "$P_AGENT" \
'MicroXRCEAgent udp4 -p 8888' C-m

tmux send-keys -t "$P_QGC" \
'~/DroneSimulator/QGroundControl-x86_64.AppImage' C-m

# 5) 最后选中 pane0
tmux select-pane -t "$P_ISAAC"
tmux attach -t "$SESSION"
