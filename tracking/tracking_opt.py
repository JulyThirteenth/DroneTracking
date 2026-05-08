import casadi as ca
import numpy as np
from dataclasses import dataclass
import datetime

from tracking.tracking_utils import Polyline3D, coerce_path_points_3d


class Dynamics:
    """
    Unified dynamics wrapper supporting multiple systems.
    """

    def __init__(self, dyn_type: str, **params):
        self.dyn_type = dyn_type.lower()
        self.params = params

        if self.dyn_type == "uav":
            self.f = self._f_uav
        else:
            raise ValueError(f"Unknown dynamics type: {dyn_type}")

    # ===============================
    # Quadrotor
    # ===============================
    @staticmethod
    def _f_uav(x, u, grad=True):
        # x = [p, v, a], u = jerk
        p_dot = x[3:6]
        v_dot = x[6:9]
        a_dot = u

        if grad:
            return ca.vertcat(p_dot, v_dot, a_dot)
        else:
            return np.concatenate(
                [
                    np.asarray(p_dot),
                    np.asarray(v_dot),
                    np.asarray(a_dot),
                ]
            )

    # ===============================
    # Integration
    # ===============================
    def rk4(self, x, u, dt, grad=True):
        f = lambda x_, u_: self.f(x_, u_, grad)

        k1 = f(x, u)
        k2 = f(x + dt / 2 * k1, u)
        k3 = f(x + dt / 2 * k2, u)
        k4 = f(x + dt * k3, u)

        return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def __call__(self, x, u, dt, grad=True, method="rk4"):
        if method == "rk4":
            return self.rk4(x, u, dt, grad)
        else:
            raise ValueError(f"Unknown integration method: {method}")


@dataclass
class OptimizerParams:
    nx: int
    nu: int
    horizon: int
    dyn_func: callable
    dt: float
    u_min: np.ndarray | None
    u_max: np.ndarray | None
    du_min: np.ndarray | None
    du_max: np.ndarray | None
    Q: np.ndarray
    R: np.ndarray
    terminal: float
    Rd: np.ndarray
    track_idx: np.ndarray


class Optimizer:
    def __init__(self, params: OptimizerParams):
        self.opti = None
        self.variables = {}
        self.parameters = {}
        self.costs = {}
        self.solver_times = []
        self.params = params

        assert self.params.Q.shape == (
            len(self.params.track_idx),
            len(self.params.track_idx),
        )
        assert self.params.R.shape == (self.params.nu, self.params.nu)
        assert self.params.Rd.shape == (self.params.nu, self.params.nu)

    def initialize_variables(self):
        self.variables["x"] = self.opti.variable(
            self.params.nx, self.params.horizon + 1
        )
        self.variables["u"] = self.opti.variable(self.params.nu, self.params.horizon)

    def initialize_parameters(self):
        self.parameters["x_init"] = self.opti.parameter(self.params.nx)
        self.parameters["u_init"] = self.opti.parameter(self.params.nu)
        self.parameters["x_target"] = self.opti.parameter(len(self.params.track_idx))
        # Per-step reference for tracked states (e.g. positions). Shape: (n_track, N+1).
        self.parameters["x_ref"] = self.opti.parameter(
            len(self.params.track_idx), self.params.horizon + 1
        )

    def add_initial_constraints(self):
        self.opti.subject_to(self.variables["x"][:, 0] == self.parameters["x_init"])

    def add_dynamics_constraints(self):
        for k in range(self.params.horizon):
            x_next = self.params.dyn_func(
                self.variables["x"][:, k],
                self.variables["u"][:, k],
                self.params.dt,
                grad=True,
            )
            self.opti.subject_to(self.variables["x"][:, k + 1] == x_next)

    def add_input_constraints(self):
        if self.params.u_min is not None and self.params.u_max is not None:
            self.opti.subject_to(
                self.opti.bounded(
                    self.params.u_min,
                    self.variables["u"],
                    self.params.u_max,
                )
            )

    def add_input_derivative_constraints(self):
        if self.params.du_min is not None and self.params.du_max is not None:
            for k in range(self.params.horizon - 1):
                du = self.variables["u"][:, k + 1] - self.variables["u"][:, k]
                self.opti.subject_to(
                    self.opti.bounded(
                        self.params.du_min * self.params.dt,
                        du,
                        self.params.du_max * self.params.dt,
                    )
                )
            du = self.variables["u"][:, 0] - self.parameters["u_init"]
            self.opti.subject_to(
                self.opti.bounded(
                    self.params.du_min * self.params.dt,
                    du,
                    self.params.du_max * self.params.dt,
                )
            )

    def add_target_cost(self):
        x = self.variables["x"]
        cost = 0
        for k in range(self.params.horizon):
            state_error = x[self.params.track_idx, k] - self.parameters["x_ref"][:, k]
            cost += ca.mtimes([state_error.T, self.params.Q, state_error])
        terminal_error = (
            x[self.params.track_idx, self.params.horizon]
            - self.parameters["x_ref"][:, self.params.horizon]
        )
        cost += self.params.terminal * ca.mtimes(
            [terminal_error.T, self.params.Q, terminal_error]
        )
        self.costs["target_cost"] = cost

    def add_input_cost(self):
        u = self.variables["u"]
        cost = 0
        for k in range(self.params.horizon):
            control = u[:, k]
            cost += ca.mtimes([control.T, self.params.R, control])
        self.costs["input_cost"] = cost

    def add_input_derivative_cost(self):
        u = self.variables["u"]
        cost = 0
        for k in range(self.params.horizon - 1):
            du = u[:, k + 1] - u[:, k]
            cost += ca.mtimes([du.T, self.params.Rd, du])
        du0 = u[:, 0] - self.parameters["u_init"]
        cost += ca.mtimes([du0.T, self.params.Rd, du0])
        self.costs["input_derivative_cost"] = cost

    def add_warmstart(self, x_ws, u_ws):
        x_ws = np.hstack([x_ws[:, 1:], x_ws[:, -1:]])
        u_ws = np.hstack([u_ws[:, 1:], u_ws[:, -1:]])
        self.opti.set_initial(self.variables["x"], x_ws)
        self.opti.set_initial(self.variables["u"], u_ws)

    def add_constraints(self):
        self.add_initial_constraints()  # x0的约束
        self.add_dynamics_constraints()  # 系统约束
        self.add_input_constraints()  # 输入约束
        self.add_input_derivative_constraints()  # 输入变化约束

    def add_costs(self):
        self.add_target_cost()  # 目标损失
        self.add_input_cost()  # 输入损失
        self.add_input_derivative_cost()  # 输入变化损失

    def setup(self):
        self.costs.clear()
        self.variables.clear()
        self.opti = ca.Opti()
        self.initialize_variables()
        self.initialize_parameters()

        # constraints
        self.add_constraints()

        # costs
        self.add_costs()

        self.opti.minimize(sum(self.costs.values()))
        self.opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,
                "print_time": 0,
            },
        )

    @staticmethod
    def _coerce_ref_traj(ref, *, n_track: int, horizon: int) -> np.ndarray:
        """
        Accept either a single target (n_track,) or a per-step reference trajectory.

        Supported shapes:
          - (n_track,)
          - (n_track, N+1) or (N+1, n_track)
          - (n_track, N)   or (N, n_track)  -> padded by repeating the last column/row
        """
        arr = np.asarray(ref, dtype=float)

        if arr.ndim == 1:
            if arr.size != n_track:
                raise ValueError(
                    f"ref must have shape ({n_track},) or a trajectory; got {arr.shape}"
                )
            return np.repeat(arr.reshape(n_track, 1), horizon + 1, axis=1)

        if arr.ndim != 2:
            raise ValueError(
                f"ref must be 1D or 2D array-like; got ndim={arr.ndim}, shape={arr.shape}"
            )

        if arr.shape == (n_track, horizon + 1):
            return arr
        if arr.shape == (horizon + 1, n_track):
            return arr.T

        # allow missing terminal reference
        if arr.shape == (n_track, horizon):
            last = arr[:, -1:].copy()
            return np.hstack([arr, last])
        if arr.shape == (horizon, n_track):
            last = arr[-1:, :].copy()
            return np.vstack([arr, last]).T

        raise ValueError(
            f"ref trajectory must be ({n_track}, {horizon+1}) / ({horizon+1}, {n_track}) "
            f"or ({n_track}, {horizon}) / ({horizon}, {n_track}); got {arr.shape}"
        )

    def solve(self, x0, u0, ref, x_ws, u_ws, log=False, *kwargs):
        opti = self.opti
        opti.set_value(self.parameters["x_init"], x0)
        opti.set_value(self.parameters["u_init"], u0)
        ref_traj = self._coerce_ref_traj(
            ref, n_track=len(self.params.track_idx), horizon=self.params.horizon
        )
        opti.set_value(self.parameters["x_ref"], ref_traj)
        # Keep `x_target` consistent (e.g. for downstream code that still reads goal).
        opti.set_value(self.parameters["x_target"], ref_traj[:, -1])

        # print(x0)
        # print(u0)
        # print(ref)

        self.add_warmstart(x_ws, u_ws)

        start_timer = datetime.datetime.now()
        opt_sol = opti.solve()
        end_timer = datetime.datetime.now()
        delta_timer = end_timer - start_timer
        self.solver_times.append(delta_timer.total_seconds())
        if log:
            print(
                f"solver time: {delta_timer.total_seconds()}, \
                    max time: {max(self.solver_times)}, \
                    mean_ time:{np.mean(self.solver_times)}",
            )
        return opt_sol.value(self.variables["x"]), opt_sol.value(self.variables["u"])


@dataclass
class MPCParams(OptimizerParams):
    v_max: np.ndarray | None
    v_min: np.ndarray | None
    a_max: np.ndarray | None
    a_min: np.ndarray | None
    v_slack_weight: float
    a_slack_weight: float


class MPC(Optimizer):
    def __init__(self, params: MPCParams):
        super().__init__(params)
        self.params: MPCParams = params

    def initialize_variables_uav(self):
        self.initialize_variables()

    def initialize_parameters_uav(self):
        self.initialize_parameters()

    def add_vel_cons(self):
        if self.params.v_min is not None and self.params.v_max is not None:
            v = self.variables["x"][3:6, :]
            slack = self.opti.variable(3, self.params.horizon + 1)
            self.opti.subject_to(ca.vec(slack) >= 0)
            self.opti.subject_to(
                ca.vec(v) >= ca.vec(self.params.v_min[:, None] - slack)
            )
            self.opti.subject_to(
                ca.vec(v) <= ca.vec(self.params.v_max[:, None] + slack)
            )
            self.costs["v_slack"] = self.params.v_slack_weight * ca.sumsqr(slack)

    def add_acc_cons(self):
        if self.params.a_min is not None and self.params.a_max is not None:
            a = self.variables["x"][6:9, :]
            slack = self.opti.variable(3, self.params.horizon + 1)
            self.opti.subject_to(ca.vec(slack) >= 0)
            self.opti.subject_to(
                ca.vec(a) >= ca.vec(self.params.a_min[:, None] - slack)
            )
            self.opti.subject_to(
                ca.vec(a) <= ca.vec(self.params.a_max[:, None] + slack)
            )
            self.costs["a_slack"] = self.params.a_slack_weight * ca.sumsqr(slack)

    def add_constraints_uav(self):
        self.add_constraints()
        self.add_vel_cons()  # 添加速度约束
        self.add_acc_cons()  # 添加加速度约束

    def setup(self):
        self.costs.clear()
        self.variables.clear()
        self.opti = ca.Opti()
        self.initialize_variables_uav()
        self.initialize_parameters_uav()

        # constraints
        self.add_constraints_uav()

        # costs
        self.add_costs()

        self.opti.minimize(sum(self.costs.values()))
        self.opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,
                "print_time": 0,
            },
        )


@dataclass
class MPCCParams(MPCParams):
    q_contour: float
    q_lag: float
    q_progress: float
    q_terminal_s: float
    vs_max: float | None
    vs_min: float | None


class MPCC(MPC):
    """
    MPCC for UAV in flatness space with a straight-line reference constructed from
    x_init (current position) and x_target (goal position).

    Reference line:
        p_r(s) = p0 + s * t,  t = (pg - p0) / ||pg - p0||,  L = ||pg - p0||
    Errors:
        e_c = (I - t t^T) (p - p0)          (contouring error vector)
        e_l = t^T (p - p0) - s             (lag error scalar)
    Progress:
        s_{k+1} = s_k + dt * v_s,k,   v_s,k >= 0
    """

    def __init__(self, params: MPCCParams):
        super().__init__(params)
        self.params: MPCCParams = params
        self._path: Polyline3D | None = None
        self._path_points: np.ndarray | None = None  # (M,3) in same frame as state p
        self._path_s: np.ndarray | None = None  # (M,)
        self._path_L: float | None = None
        self._path_interp = None

    # --------------------------
    # Variables / Parameters
    # --------------------------
    def initialize_variables_uav(self):
        super().initialize_variables_uav()
        self.variables["s"] = self.opti.variable(1, self.params.horizon + 1)
        self.variables["vs"] = self.opti.variable(1, self.params.horizon)

    def initialize_parameters_uav(self):
        super().initialize_parameters_uav()
        self.parameters["s_init"] = self.opti.parameter(1)

    # --------------------------
    # Constraints
    # --------------------------
    def add_progress_constraints(self):
        s = self.variables["s"]
        vs = self.variables["vs"]
        dt = float(self.params.dt)
        eps = 1e-6

        # progress initialization (can also be warm-started)
        self.opti.subject_to(s[0, 0] == self.parameters["s_init"][0])

        # progress dynamics
        for k in range(self.params.horizon):
            self.opti.subject_to(s[0, k + 1] == s[0, k] + dt * vs[0, k])

        # monotone progress
        self.opti.subject_to(vs >= 0)

        # keep progress within the active reference length.
        self.opti.subject_to(s >= 0)
        if self._path_interp is None:
            d = self.parameters["x_target"][0:3] - self.parameters["x_init"][0:3]
            line_length = ca.sqrt(ca.dot(d, d) + eps)
            self.opti.subject_to(s <= line_length)
        elif self._path_L is not None:
            self.opti.subject_to(s <= float(self._path_L))

        # optional bounds (if you set params.vs_min / params.vs_max)
        vs_min = getattr(self.params, "vs_min", None)
        vs_max = getattr(self.params, "vs_max", None)
        if (vs_min is not None) and (vs_max is not None):
            self.opti.subject_to(self.opti.bounded(vs_min, vs, vs_max))

    def add_constraints_uav(self):
        # keep all existing constraints from MPC: init + dynamics + u/du + v/a
        super().add_constraints_uav()
        # add MPCC progress constraints
        self.add_progress_constraints()

    # --------------------------
    # MPCC Cost
    # --------------------------
    def set_path(self, path_points):
        pts = coerce_path_points_3d(path_points)
        if pts is None:
            raise ValueError("set_path expects a 2D path array (M,3)/(3,M)")
        poly = Polyline3D.from_points(pts, dedupe=True)
        pts = poly.points
        s_grid = poly.s
        L = poly.length

        # Use a smooth interpolant when possible; fall back to linear for short paths.
        interp_type = "bspline" if pts.shape[0] >= 4 else "linear"
        self._path = poly
        self._path_points = pts
        self._path_s = s_grid
        self._path_L = L
        self._path_interp = ca.interpolant(
            "path_xyz",
            interp_type,
            [s_grid],
            pts.T,
        )

    def add_mpcc_cost(self):
        x = self.variables["x"]
        s = self.variables["s"]
        vs = self.variables["vs"]

        # positions over horizon
        p = x[0:3, :]  # (3, N+1)

        # weights (override by setting these attributes on params)
        q_c = float(getattr(self.params, "q_contour", 20.0))  # contouring
        q_l = float(getattr(self.params, "q_lag", 2.0))  # lag
        q_s = float(getattr(self.params, "q_progress", 5.0))  # encourage progress
        q_term = float(getattr(self.params, "q_terminal_s", 50.0))  # reach end

        cost = 0
        N = self.params.horizon
        eps = 1e-6
        I3 = ca.DM_eye(3)

        if self._path_interp is None:
            # Straight-line MPCC using current position as p0 and x_target as goal.
            p0 = self.parameters["x_init"][0:3]  # (3,)
            pg = self.parameters["x_target"][0:3]  # (3,)

            d = pg - p0
            L = ca.sqrt(ca.dot(d, d) + eps)  # scalar
            t = d / L  # (3,)
            Pperp = I3 - ca.mtimes(t, t.T)  # 3x3

            for k in range(N):
                pk = p[:, k]
                dp = pk - p0

                ec = ca.mtimes(Pperp, dp)
                el = ca.mtimes(t.T, dp) - s[0, k]

                cost += q_c * ca.sumsqr(ec) + q_l * ca.sumsqr(el)
                cost += -q_s * vs[0, k]

            cost += q_term * (L - s[0, N]) ** 2
        else:
            # Curve MPCC using a fixed path interpolant p_r(s) built from waypoints.
            L = float(self._path_L)
            for k in range(N):
                sk = s[0, k]
                pk = p[:, k]

                pr = self._path_interp(sk)  # (3,1)
                dpr = ca.jacobian(pr, sk)  # (3,1)
                t = dpr / (ca.norm_2(dpr) + eps)  # (3,1)
                Pperp = I3 - ca.mtimes(t, t.T)

                dp = pk - pr
                ec = ca.mtimes(Pperp, dp)
                el = ca.mtimes(t.T, dp)

                cost += q_c * ca.sumsqr(ec) + q_l * ca.sumsqr(el)
                cost += -q_s * vs[0, k]

            cost += q_term * (float(L) - s[0, N]) ** 2

        self.costs["mpcc_cost"] = cost

    # --------------------------
    # Costs (override)
    # --------------------------
    def add_costs(self):
        # IMPORTANT: do NOT call super().add_costs() because it includes add_target_cost()
        self.add_mpcc_cost()
        self.add_input_cost()
        self.add_input_derivative_cost()

    # --------------------------
    # Setup / Solve
    # --------------------------
    def setup(self):
        self.costs.clear()
        self.variables.clear()
        self.opti = ca.Opti()

        self.initialize_variables_uav()
        self.initialize_parameters_uav()

        self.add_constraints_uav()
        self.add_costs()

        self.opti.minimize(sum(self.costs.values()))
        self.opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,
                "print_time": 0,
            },
        )

    def solve(
        self,
        x0,
        u0,
        target,  # (3,) goal position OR a polyline path (M,3)/(3,M)
        x_ws,
        u_ws,
        s_ws=None,  # optional warmstart for s
        vs_ws=None,  # optional warmstart for vs
        log: bool = False,
    ):
        path_pts = coerce_path_points_3d(target)
        if path_pts is not None:
            path_pts = Polyline3D.from_points(path_pts, dedupe=True).points
            rebuild = False
            if self._path_points is None:
                rebuild = True
            else:
                if self._path_points.shape != path_pts.shape or not np.allclose(
                    self._path_points, path_pts, atol=1e-9, rtol=0.0
                ):
                    rebuild = True
            if rebuild:
                self.set_path(path_pts)
                self.setup()

        self.last_solve_inputs = {
            "x0": np.array(x0, dtype=float, copy=True),
            "u0": np.array(u0, dtype=float, copy=True),
            "target": np.array(target, dtype=float, copy=True),
            "x_ws": np.array(x_ws, dtype=float, copy=True),
            "u_ws": np.array(u_ws, dtype=float, copy=True),
        }
        opti = self.opti

        # set parameters (reuse existing ones)
        opti.set_value(self.parameters["x_init"], x0)
        opti.set_value(self.parameters["u_init"], u0)
        if path_pts is None:
            opti.set_value(self.parameters["x_target"], target)
            opti.set_value(self.parameters["s_init"], np.array([0.0], dtype=float))
        else:
            # keep x_target consistent (goal = last waypoint)
            opti.set_value(self.parameters["x_target"], path_pts[-1])
            if self._path is None:
                raise RuntimeError("Path not initialized; call set_path() first.")
            s_init = self._path.closest_s(np.asarray(x0, dtype=float).reshape(-1)[0:3])
            opti.set_value(self.parameters["s_init"], np.array([s_init], dtype=float))

        # warmstart x/u
        self.add_warmstart(x_ws, u_ws)

        # warmstart s / vs (optional)
        if s_ws is not None:
            s_ws = np.hstack((s_ws[:, 1:], s_ws[:, -1:]))
            opti.set_initial(self.variables["s"], s_ws)
        if vs_ws is not None:
            vs_ws = np.hstack((vs_ws[:, 1:], vs_ws[:, -1:]))
            opti.set_initial(self.variables["vs"], vs_ws)

        start_timer = datetime.datetime.now()
        opt_sol = opti.solve()
        end_timer = datetime.datetime.now()
        delta_timer = end_timer - start_timer
        self.solver_times.append(delta_timer.total_seconds())
        # print(
        #     f"solver time: {delta_timer.total_seconds()}, "
        #     f"max time: {max(self.solver_times)}, "
        #     f"mean_ time:{np.mean(self.solver_times)}"
        # )

        return (
            opt_sol.value(self.variables["x"]),
            opt_sol.value(self.variables["u"]),
            opt_sol.value(self.variables["s"]),
            opt_sol.value(self.variables["vs"]),
        )
