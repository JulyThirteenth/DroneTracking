#!/usr/bin/env bash

SESSION="dronesim"

# 已存在就直接进入
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux attach -t "$SESSION"
  exit 0
fi

# 新建 session
tmux new-session -d -s "$SESSION" -n main "bash"

# 1) 先上下分割：下方占 10%
tmux split-window -v -p 10 -t "$SESSION":0.0

# 2) 在下方做三列横向并排
tmux select-pane -t "$SESSION":0.1
tmux split-window -h -t "$SESSION":0.1
tmux split-window -h -t "$SESSION":0.2

# 3) Pane 0：进入同目录下的 simdrone_env.sh 环境（mpcc + ROS）
# 说明：用 bash -lc 确保 conda 函数可用，然后在里面 source 脚本并进入交互 shell
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_SH="$SCRIPT_DIR/simdrone_env.sh"

tmux send-keys -t "$SESSION":0.0 \
"cd \"$SCRIPT_DIR\" && source \"$ENV_SH\" && clear" C-m


# 4) 下方三个 pane 执行命令
tmux send-keys -t "$SESSION":0.1 \
'isaac_run ~/DroneSimulator/PegasusSimulator/examples/drone_racing/isaacsim/sim_single.py' C-m

tmux send-keys -t "$SESSION":0.2 \
'MicroXRCEAgent udp4 -p 8888' C-m

tmux send-keys -t "$SESSION":0.3 \
'~/DroneSimulator/QGroundControl-x86_64.AppImage' C-m

# 5) 最后选中 pane0
tmux select-pane -t "$SESSION":0.1
tmux attach -t "$SESSION"