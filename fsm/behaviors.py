"""Behavior layer for the drone racing FSM.

`DroneBehaviors` owns the low-level tracker and PX4 bridge. The FSM node feeds it:
  - vehicle telemetry (`VehicleState`)
  - path / reference trajectory topics
  - state transitions (via `on_enter`)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class VehicleState:
    """Vehicle state in ENU coordinates."""

    position_enu: np.ndarray
    velocity_enu: np.ndarray
    accel_enu: np.ndarray
    yaw_enu: float


def path_msg_points_enu(msg: Any) -> np.ndarray:
    """Extract ENU points from a `nav_msgs/Path`-like message."""
    poses = getattr(msg, "poses", None) or []
    return np.array(
        [[ps.pose.position.x, ps.pose.position.y, ps.pose.position.z] for ps in poses],
        dtype=float,
    )


class DroneBehaviors:
    """Per-state behaviors that translate FSM state into tracking targets."""

    def __init__(
        self,
        *,
        bridge: Any,
        tracker: Any,
        logger: Any | None = None,
        takeoff_speed_mps: float = 0.67,
        init_yaw_enu: float = 0.0,
    ):
        self._bridge = bridge
        self._tracker = tracker
        self._fsm_log = logger
        self._mode = str(getattr(tracker, "controller", "mpc")).lower().strip()
        self._horizon = int(getattr(tracker, "horizon", 10))
        self._tracker_dt = float(getattr(tracker, "dt", 0.1))
        self._takeoff_speed_mps = max(float(takeoff_speed_mps), 1e-3)
        self._init_yaw_enu = float(init_yaw_enu)

        self._state: VehicleState | None = None
        self._fsm_state_name: str = "preflight"
        self._path_pts: np.ndarray | None = None  # (M, 3) for MPCC
        self._ref_traj: np.ndarray | None = None  # (3, N+1) for MPC
        self._start_pt: np.ndarray | None = None  # (3,)
        self._hold_point_enu: np.ndarray | None = None
        self._hold_key: str | None = None
        self._yaw_cmd_enu: float = -np.pi / 2.0
        self._disengaged: bool = False  # stop publishing offboard setpoints (AUTO.LAND)

    def _set_hold(self, point_enu: np.ndarray, *, key: str) -> np.ndarray:
        self._hold_point_enu = np.asarray(point_enu, dtype=float).reshape(3).copy()
        self._hold_key = str(key)
        return self._hold_point_enu

    def _hold_target_once(self, *, key: str, point_enu: np.ndarray) -> np.ndarray:
        if self._hold_key != key or self._hold_point_enu is None:
            return self._set_hold(point_enu, key=key)
        return self._hold_point_enu

    def _hold_current_once(self, *, key: str) -> np.ndarray:
        if self._state is None:
            return np.zeros(3, dtype=float)
        return self._hold_target_once(key=key, point_enu=self._state.position_enu)

    def _clear_hold(self) -> None:
        self._hold_point_enu = None
        self._hold_key = None

    def _clear_tracking_hold(self) -> None:
        if self._hold_key is not None and self._hold_key.startswith("tracking:"):
            self._clear_hold()

    def update_vehicle_state(self, state: VehicleState) -> None:
        self._state = state

    def update_yaw_cmd_enu(self, yaw_cmd_enu: float) -> None:
        if self._fsm_state_name == "hover_start":
            self._yaw_cmd_enu = self._init_yaw_enu
            return
        self._yaw_cmd_enu = float(yaw_cmd_enu)

    def update_path(self, path_pts_enu: np.ndarray) -> None:
        pts = np.asarray(path_pts_enu, dtype=float).reshape(-1, 3)
        if int(pts.shape[0]) < 1:
            return

        self._path_pts = pts
        if self._start_pt is None:
            self._start_pt = pts[0].copy()
            print(f"Update path: Set start_pt to enu {self._start_pt}")

        if self._mode == "mpcc" and int(pts.shape[0]) >= 2:
            self._tracker.set_polyline(pts)

    def update_ref_traj(self, ref_traj_enu: np.ndarray) -> None:
        pts = np.asarray(ref_traj_enu, dtype=float).reshape(-1, 3)
        if int(pts.shape[0]) < 2:
            return

        if self._start_pt is None:
            self._start_pt = pts[0].copy()
            print(f"Update ref_traj: Set start_pt to enu {self._start_pt}")

        n = self._horizon + 1
        self._ref_traj = pts[:n].T  # (3, N+1)

    def tracking_terminal_metrics(self) -> tuple[bool, float, float]:
        """Return whether the tracking terminal target is valid plus distance/speed."""
        if self._state is None:
            return False, float("inf"), float("inf")

        target_enu: np.ndarray | None = None
        if self._mode == "mpcc":
            if self._path_pts is not None and int(self._path_pts.shape[0]) >= 1:
                target_enu = np.asarray(self._path_pts[-1], dtype=float).reshape(3)
        else:
            if self._ref_traj is not None and int(self._ref_traj.shape[1]) >= 1:
                target_enu = np.asarray(self._ref_traj[:, -1], dtype=float).reshape(3)

        if target_enu is None:
            return False, float("inf"), float("inf")

        pos_enu = np.asarray(self._state.position_enu, dtype=float).reshape(3)
        vel_enu = np.asarray(self._state.velocity_enu, dtype=float).reshape(3)
        distance_m = float(np.linalg.norm(target_enu - pos_enu))
        speed_mps = float(np.linalg.norm(vel_enu))
        return True, distance_m, speed_mps

    def on_enter(self, state: str, event_name: str) -> None:
        """Handle FSM state-entry actions."""
        self._fsm_state_name = str(state)
        self._tracker.reset_warmstart()
        self._clear_hold()
        if self._fsm_log is not None:
            try:
                self._fsm_log.log_event(state=str(state), event=str(event_name))
            except Exception:
                pass

        if state == "ready":
            self._disengaged = False
            self._bridge.send_vehicle_command(176, 1.0, 6.0)  # OFFBOARD
            return

        if state == "hover_start" and event_name == "cmd_takeoff":
            self._disengaged = False
            self._bridge.send_vehicle_command(176, 1.0, 6.0)  # OFFBOARD
            self._bridge.send_vehicle_command(400, 1.0, 0.0)  # ARM
            return

        if state == "preflight" and event_name == "cmd_land":
            self._disengaged = True
            self._bridge.send_vehicle_command(21, 0.0, 0.0)  # NAV_LAND
            self._path_pts = None
            self._ref_traj = None
            self._start_pt = None

    def tick(self, fsm_state: str, dt: float) -> None:
        """Advance the behavior loop for the current FSM state."""
        if self._state is None or self._disengaged or fsm_state == "preflight":
            return

        self._bridge.publish_offboard_mode()

        hold_enu = self._hold_position(fsm_state)
        ref_traj = self._mpc_reference(fsm_state, hold_enu)

        if self._mode != "mpc" and hold_enu is not None:
            self._tracker.set_polyline(np.asarray(hold_enu, dtype=float).reshape(1, 3))

        p_cmd, q_cmd, r_cmd, thrust, _ = self._tracker.step(
            self._state.position_enu,
            self._state.velocity_enu,
            self._state.accel_enu,
            self._state.yaw_enu,
            float(dt),
            yaw_cmd_enu=self._yaw_cmd_enu,
            ref_traj_enu=ref_traj,
            log_solver=False,
        )
        self._bridge.publish_rates_setpoint(p_cmd, q_cmd, r_cmd, thrust)

        if self._fsm_log is not None:
            try:
                from fsm_log import TickData

                dbg = getattr(self._tracker, "last_debug", {}) or {}
                self._fsm_log.log_tick(
                    TickData(
                        fsm_state=str(fsm_state),
                        pos_enu=np.asarray(self._state.position_enu, dtype=float),
                        vel_enu=np.asarray(self._state.velocity_enu, dtype=float),
                        acc_enu=np.asarray(self._state.accel_enu, dtype=float),
                        yaw_enu=float(self._state.yaw_enu),
                        p_cmd=float(p_cmd),
                        q_cmd=float(q_cmd),
                        r_cmd=float(r_cmd),
                        thrust_cmd=float(thrust),
                        jerk_cmd_enu=dbg.get("jerk_cmd_enu", None),
                        acc_cmd_enu=dbg.get("acc_cmd_enu", None),
                        jerk_cmd_ned=dbg.get("jerk_cmd_ned", None),
                        acc_cmd_ned=dbg.get("acc_cmd_ned", None),
                        yaw_cmd_enu=dbg.get("yaw_cmd_enu", None),
                        yaw_rate_cmd_enu=dbg.get("yaw_rate_cmd_enu", None),
                        yaw_cmd_ned=dbg.get("yaw_cmd_ned", None),
                        yaw_rate_cmd_ned=dbg.get("yaw_rate_cmd_ned", None),
                    )
                )
            except Exception:
                pass

    def _hold_position(self, fsm_state: str) -> np.ndarray | None:
        """Return an ENU point to hold, or None if no hold override is needed."""
        if self._state is None:
            return None

        if fsm_state in {"preflight", "ready", "hover"}:
            return self._hold_current_once(key=fsm_state)

        if fsm_state == "hover_start":
            if self._start_pt is not None:
                return self._hold_target_once(
                    key="hover_start:start", point_enu=self._start_pt
                )

            target = np.asarray(self._state.position_enu, dtype=float).reshape(3).copy()
            target[2] = float(target[2] + 1.0)
            return self._hold_target_once(
                key="hover_start:takeoff_1m", point_enu=target
            )

        if fsm_state == "return_hover":
            if self._start_pt is not None:
                return self._hold_target_once(
                    key="return_hover:start", point_enu=self._start_pt
                )

            return self._hold_current_once(key="return_hover:current")

        if fsm_state == "tracking":
            if self._mode == "mpcc":
                if self._path_pts is None or int(self._path_pts.shape[0]) < 1:
                    return self._hold_current_once(
                        key="tracking:mpcc:hold_current_no_path"
                    )
                if int(self._path_pts.shape[0]) < 2:
                    return self._hold_target_once(
                        key="tracking:mpcc:hold_last_singleton",
                        point_enu=self._path_pts[-1],
                    )
                self._clear_tracking_hold()
                return None

            if self._ref_traj is None:
                return self._hold_current_once(key="tracking:mpc:hold_current_no_ref")
            self._clear_tracking_hold()
            return None

        return self._hold_current_once(key="unknown")

    def _mpc_reference(
        self, fsm_state: str, hold_enu: np.ndarray | None
    ) -> np.ndarray | None:
        """Return an MPC reference trajectory, or None for MPCC mode."""
        if self._mode != "mpc":
            return None

        if fsm_state == "tracking" and self._ref_traj is not None:
            return self._ref_traj

        if hold_enu is None:
            return None

        p_hold = np.asarray(hold_enu, dtype=float).reshape(3)
        if self._state is not None:
            p_now = np.asarray(self._state.position_enu, dtype=float).reshape(3)
            dist = float(np.linalg.norm(p_hold - p_now))
            if np.isfinite(dist) and dist > 1e-3:
                if fsm_state == "hover_start":
                    step_s = self._takeoff_speed_mps * self._tracker_dt
                    samples = step_s * np.arange(self._horizon + 1, dtype=float)
                    progress = np.clip(samples / dist, 0.0, 1.0)
                    delta = (p_hold - p_now).reshape(3, 1)
                    return p_now.reshape(3, 1) + delta * progress.reshape(1, -1)

                alpha = np.linspace(0.0, 1.0, self._horizon + 1, dtype=float)
                delta = (p_hold - p_now).reshape(3, 1)
                return p_now.reshape(3, 1) + delta * alpha.reshape(1, -1)

        p = p_hold.reshape(3, 1)
        return np.repeat(p, self._horizon + 1, axis=1)
