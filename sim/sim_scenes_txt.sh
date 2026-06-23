#!/usr/bin/env bash
set -Eeuo pipefail

SESSION="${SIMDRONE_SESSION:-dronesim}"

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &&
    pwd
)"
PROJECT_DIR="$(
    cd -- "$SCRIPT_DIR/.." &&
    pwd
)"

XRCE_AGENT="${XRCE_AGENT:-MicroXRCEAgent}"
XRCE_PORT="${XRCE_PORT:-8888}"
QGC_PATH="${QGC_PATH:-$HOME/DroneSimulator/QGroundControl-x86_64.AppImage}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux attach-session -t "$SESSION"
    exit 0
fi

command -v tmux >/dev/null 2>&1 || {
    echo "ERROR: tmux not found"
    exit 1
}

command -v "$XRCE_AGENT" >/dev/null 2>&1 || {
    echo "ERROR: MicroXRCEAgent not found"
    exit 1
}

printf -v SIM_ARGS ' %q' "$@"

tmux new-session \
    -d \
    -s "$SESSION" \
    -n simulation

P_ISAAC="$(
    tmux display-message \
        -p \
        -t "$SESSION":0.0 \
        '#{pane_id}'
)"

P_AGENT="$(
    tmux split-window \
        -h \
        -p 30 \
        -t "$P_ISAAC" \
        -P \
        -F '#{pane_id}'
)"

tmux send-keys \
    -t "$P_ISAAC" \
    "cd $(printf '%q' "$PROJECT_DIR") && \
source ~/.bashrc && \
unset LD_PRELOAD AMENT_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH && \
isaac_run sim/sim_scenes_txt.py${SIM_ARGS}" \
    C-m

tmux send-keys \
    -t "$P_AGENT" \
    "env -u LD_PRELOAD -u LD_LIBRARY_PATH \
$(printf '%q' "$XRCE_AGENT") udp4 -p $(printf '%q' "$XRCE_PORT")" \
    C-m

if [[ -n "$QGC_PATH" ]]; then
    if [[ ! -x "$QGC_PATH" ]]; then
        echo "ERROR: QGC_PATH is not executable: $QGC_PATH"
        tmux kill-session -t "$SESSION"
        exit 1
    fi

    P_QGC="$(
        tmux split-window \
            -v \
            -p 50 \
            -t "$P_AGENT" \
            -P \
            -F '#{pane_id}'
    )"

    tmux send-keys \
        -t "$P_QGC" \
        "$(printf '%q' "$QGC_PATH")" \
        C-m
fi

tmux select-pane -t "$P_ISAAC"
tmux attach-session -t "$SESSION"