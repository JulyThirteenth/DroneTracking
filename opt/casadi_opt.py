import casadi as ca
import numpy as np
from dataclasses import dataclass
import datetime


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
    terminal_v_weight: float
    terminal_a_weight: float


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
            state_error = x[self.params.track_idx, k] - self.parameters["x_target"]
            cost += ca.mtimes([state_error.T, self.params.Q, state_error])
        terminal_error = (
            x[self.params.track_idx, self.params.horizon] - self.parameters["x_target"]
        )
        cost += self.params.terminal * ca.mtimes(
            [terminal_error.T, self.params.Q, terminal_error]
        )
        # penalize velocity/acceleration for smooth stop
        v_term = x[3:6, self.params.horizon]
        a_term = x[6:9, self.params.horizon]
        w_v = float(getattr(self.params, "terminal_v_weight", 5.0))
        w_a = float(getattr(self.params, "terminal_a_weight", 2.0))
        cost += w_v * ca.sumsqr(v_term) + w_a * ca.sumsqr(a_term)
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

    def solve(self, x0, u0, target, x_ws, u_ws, log=False, *kwargs):
        opti = self.opti
        opti.set_value(self.parameters["x_init"], x0)
        opti.set_value(self.parameters["u_init"], u0)
        opti.set_value(self.parameters["x_target"], target)

        # print(x0)
        # print(u0)
        # print(target)

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
class ChaserParams(OptimizerParams):
    v_max: np.ndarray | None
    v_min: np.ndarray | None
    a_max: np.ndarray | None
    a_min: np.ndarray | None
    v_slack_weight: float
    a_slack_weight: float


class Chaser(Optimizer):
    def __init__(self, params: ChaserParams):
        super().__init__(params)
        self.params: ChaserParams = params

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
class TrackerParams(ChaserParams):
    q_contour: float
    q_lag: float
    q_progress: float
    q_terminal_s: float
    vs_max: float | None
    vs_min: float | None


class Tracker(Chaser):
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

    def __init__(self, params: TrackerParams):
        super().__init__(params)
        self.params: TrackerParams = params

    # --------------------------
    # Variables / Parameters
    # --------------------------
    def initialize_variables_uav(self):
        super().initialize_variables_uav()
        self.variables["s"] = self.opti.variable(1, self.params.horizon + 1)
        self.variables["vs"] = self.opti.variable(1, self.params.horizon)

    # keep parameters from parent: x_init, u_init, x_target

    # --------------------------
    # Constraints
    # --------------------------
    def add_progress_constraints(self):
        s = self.variables["s"]
        vs = self.variables["vs"]
        dt = float(self.params.dt)

        # progress initialization (can also be warm-started)
        self.opti.subject_to(s[0, 0] == 0.0)

        # progress dynamics
        for k in range(self.params.horizon):
            self.opti.subject_to(s[0, k + 1] == s[0, k] + dt * vs[0, k])

        # monotone progress
        self.opti.subject_to(vs >= 0)

        # optional bounds (if you set params.vs_min / params.vs_max)
        vs_min = getattr(self.params, "vs_min", None)
        vs_max = getattr(self.params, "vs_max", None)
        if (vs_min is not None) and (vs_max is not None):
            self.opti.subject_to(self.opti.bounded(vs_min, vs, vs_max))

    def add_constraints_uav(self):
        # keep all existing constraints from tracker: init + dynamics + u/du + v/a
        super().add_constraints_uav()
        # add MPCC progress constraints
        self.add_progress_constraints()

    # --------------------------
    # MPCC Cost
    # --------------------------
    def add_mpcc_cost(self):
        x = self.variables["x"]
        s = self.variables["s"]
        vs = self.variables["vs"]

        # positions over horizon
        p = x[0:3, :]  # (3, N+1)

        # line endpoints from existing parameters
        # NOTE: x_target length is len(track_idx). For UAV tracking you typically pass target position (3,).
        p0 = self.parameters["x_init"][0:3]  # (3,)
        pg = self.parameters["x_target"][0:3]  # (3,)

        d = pg - p0
        eps = 1e-6
        L = ca.sqrt(ca.dot(d, d) + eps)  # scalar
        t = d / L  # (3,)

        I3 = ca.DM_eye(3)
        Pperp = I3 - ca.mtimes(t, t.T)  # 3x3

        # weights (override by setting these attributes on params)
        q_c = float(getattr(self.params, "q_contour", 20.0))  # contouring
        q_l = float(getattr(self.params, "q_lag", 2.0))  # lag
        q_s = float(getattr(self.params, "q_progress", 5.0))  # encourage progress
        q_term = float(getattr(self.params, "q_terminal_s", 50.0))  # reach end

        cost = 0
        N = self.params.horizon
        for k in range(N):
            pk = p[:, k]
            dp = pk - p0

            # contouring error vector (3x1)
            ec = ca.mtimes(Pperp, dp)

            # lag error scalar
            el = ca.mtimes(t.T, dp) - s[0, k]

            cost += q_c * ca.sumsqr(ec) + q_l * ca.sumsqr(el)
            cost += -q_s * vs[0, k]  # maximize forward progress

        # terminal: encourage s_N close to segment length L (i.e., reach goal)
        cost += q_term * (L - s[0, N]) ** 2

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
        target,  # expected: (3,) goal position for UAV
        x_ws,
        u_ws,
        s_ws=None,  # optional warmstart for s
        vs_ws=None,  # optional warmstart for vs
    ):
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
        opti.set_value(self.parameters["x_target"], target)

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
