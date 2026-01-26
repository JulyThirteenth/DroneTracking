import time
from dataclasses import dataclass
import numpy as np
from utils.utils import uav_jerk_discrete_matrices
from opt.casadi_opt import ChaserParams

try:
    import osqp  # noqa: F401
    import scipy.sparse as sp  # noqa: F401
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("ChaserOSQP requires `osqp` and `scipy`.") from exc


@dataclass(frozen=True)
class _Idx:
    nx: int
    nu: int
    N: int
    x0: int
    u0: int
    sv0: int | None
    sa0: int | None

    @property
    def nvar(self) -> int:
        n = self.u0 + self.nu * self.N
        if self.sv0 is not None:
            n = self.sv0 + 3 * (self.N + 1)
        if self.sa0 is not None:
            n = self.sa0 + 3 * (self.N + 1)
        return n

    def x(self, k: int) -> slice:
        s = self.x0 + self.nx * k
        return slice(s, s + self.nx)

    def u(self, k: int) -> slice:
        s = self.u0 + self.nu * k
        return slice(s, s + self.nu)

    def sv(self, k: int) -> slice:
        assert self.sv0 is not None
        s = self.sv0 + 3 * k
        return slice(s, s + 3)

    def sa(self, k: int) -> slice:
        assert self.sa0 is not None
        s = self.sa0 + 3 * k
        return slice(s, s + 3)


class ChaserOSQP:
    """
    Solve the Chaser MPC as a convex QP using the python `osqp` solver.
    """

    def __init__(self, params: ChaserParams):
        self.params = params
        self.solver_times: list[float] = []

        self._idx: _Idx | None = None
        self._solver = None
        self._P = None
        self._A = None

        self._l_base: np.ndarray | None = None
        self._u_base: np.ndarray | None = None
        self._l_var: np.ndarray | None = None
        self._u_var: np.ndarray | None = None

        self._rows_x0: slice | None = None
        self._rows_du0: slice | None = None
        self._rows_base: int = 0

    def setup(self):
        p = self.params
        if p.nx != 9 or p.nu != 3:
            raise ValueError("ChaserOSQP supports nx=9, nu=3 (UAV p,v,a + jerk).")

        N = int(p.horizon)
        nx = int(p.nx)
        nu = int(p.nu)

        have_v = (p.v_min is not None) and (p.v_max is not None)
        have_a = (p.a_min is not None) and (p.a_max is not None)

        x0 = 0
        u0 = nx * (N + 1)
        sv0 = u0 + nu * N if have_v else None
        sa0 = (sv0 + 3 * (N + 1)) if have_v else (u0 + nu * N)
        sa0 = sa0 if have_a else None
        idx = _Idx(nx=nx, nu=nu, N=N, x0=x0, u0=u0, sv0=sv0, sa0=sa0)
        nvar = idx.nvar

        # --------------------
        # Objective: 0.5 z' P z + q' z
        # --------------------
        P = np.zeros((nvar, nvar), dtype=float)
        track_idx = np.asarray(p.track_idx, dtype=int)
        Q = np.asarray(p.Q, dtype=float)
        R = np.asarray(p.R, dtype=float)
        Rd = np.asarray(p.Rd, dtype=float)

        for k in range(N):
            xk = idx.x(k)
            P[np.ix_(xk.start + track_idx, xk.start + track_idx)] += 2.0 * Q
        xN = idx.x(N)
        P[np.ix_(xN.start + track_idx, xN.start + track_idx)] += (
            2.0 * float(p.terminal) * Q
        )

        w_v_term = float(getattr(p, "terminal_v_weight", 5.0))
        w_a_term = float(getattr(p, "terminal_a_weight", 2.0))
        P[xN.start + 3 : xN.start + 6, xN.start + 3 : xN.start + 6] += (
            2.0 * w_v_term * np.eye(3)
        )
        P[xN.start + 6 : xN.start + 9, xN.start + 6 : xN.start + 9] += (
            2.0 * w_a_term * np.eye(3)
        )

        for k in range(N):
            uk = idx.u(k)
            P[uk, uk] += 2.0 * R

        # smoothness: (u_{k+1}-u_k)' Rd (u_{k+1}-u_k)
        for k in range(N - 1):
            uk = idx.u(k)
            uk1 = idx.u(k + 1)
            P[uk, uk] += 2.0 * Rd
            P[uk1, uk1] += 2.0 * Rd
            P[uk, uk1] += -2.0 * Rd
            P[uk1, uk] += -2.0 * Rd
        # smoothness to previous input: (u0-u_init)' Rd (u0-u_init)
        P[idx.u(0), idx.u(0)] += 2.0 * Rd

        if have_v:
            w = float(p.v_slack_weight)
            for k in range(N + 1):
                sk = idx.sv(k)
                P[sk, sk] += 2.0 * w * np.eye(3)
        if have_a:
            w = float(p.a_slack_weight)
            for k in range(N + 1):
                sk = idx.sa(k)
                P[sk, sk] += 2.0 * w * np.eye(3)

        # --------------------
        # Constraints: l <= A z <= u
        # --------------------
        A_d, B_d = uav_jerk_discrete_matrices(p.dt)

        rows: list[np.ndarray] = []
        l: list[np.ndarray] = []
        u: list[np.ndarray] = []

        def add(Ak: np.ndarray, lk: np.ndarray, uk: np.ndarray):
            rows.append(Ak)
            l.append(lk)
            u.append(uk)

        # x0 == x_init (filled at solve via bounds)
        A0 = np.zeros((nx, nvar), dtype=float)
        A0[:, idx.x(0)] = np.eye(nx)
        add(A0, np.zeros(nx), np.zeros(nx))
        rows_x0 = slice(0, nx)

        # dynamics: x_{k+1} - A x_k - B u_k == 0
        for k in range(N):
            Ak = np.zeros((nx, nvar), dtype=float)
            Ak[:, idx.x(k + 1)] = np.eye(nx)
            Ak[:, idx.x(k)] = -A_d
            Ak[:, idx.u(k)] = -B_d
            add(Ak, np.zeros(nx), np.zeros(nx))

        rows_du0 = None
        if (p.du_min is not None) and (p.du_max is not None):
            du_min = np.asarray(p.du_min, dtype=float).reshape(nu)
            du_max = np.asarray(p.du_max, dtype=float).reshape(nu)
            lb_du = float(p.dt) * du_min
            ub_du = float(p.dt) * du_max

            for k in range(N - 1):
                Ak = np.zeros((nu, nvar), dtype=float)
                Ak[:, idx.u(k + 1)] = np.eye(nu)
                Ak[:, idx.u(k)] = -np.eye(nu)
                add(Ak, lb_du, ub_du)

            # u0 - u_init in [dt*du_min, dt*du_max]  <=>  u0 in [u_init+..., u_init+...]
            Ak = np.zeros((nu, nvar), dtype=float)
            Ak[:, idx.u(0)] = np.eye(nu)
            start = sum(r.shape[0] for r in rows)
            add(Ak, lb_du, ub_du)
            rows_du0 = slice(start, start + nu)

        if have_v:
            v_min = np.asarray(p.v_min, dtype=float).reshape(3)
            v_max = np.asarray(p.v_max, dtype=float).reshape(3)
            for k in range(N + 1):
                xk = idx.x(k)

                # v + s >= v_min  (same as in chaser.py: v >= v_min - s)
                Ak = np.zeros((3, nvar), dtype=float)
                Ak[:, xk.start + 3 : xk.start + 6] = np.eye(3)
                Ak[:, idx.sv(k)] = np.eye(3)
                add(Ak, v_min, np.full(3, np.inf))

                # v - s <= v_max  (same as in chaser.py: v <= v_max + s)
                Ak = np.zeros((3, nvar), dtype=float)
                Ak[:, xk.start + 3 : xk.start + 6] = np.eye(3)
                Ak[:, idx.sv(k)] = -np.eye(3)
                add(Ak, np.full(3, -np.inf), v_max)

        if have_a:
            a_min = np.asarray(p.a_min, dtype=float).reshape(3)
            a_max = np.asarray(p.a_max, dtype=float).reshape(3)
            for k in range(N + 1):
                xk = idx.x(k)

                Ak = np.zeros((3, nvar), dtype=float)
                Ak[:, xk.start + 6 : xk.start + 9] = np.eye(3)
                Ak[:, idx.sa(k)] = np.eye(3)
                add(Ak, a_min, np.full(3, np.inf))

                Ak = np.zeros((3, nvar), dtype=float)
                Ak[:, xk.start + 6 : xk.start + 9] = np.eye(3)
                Ak[:, idx.sa(k)] = -np.eye(3)
                add(Ak, np.full(3, -np.inf), a_max)

        A_base = np.vstack(rows) if rows else np.zeros((0, nvar), dtype=float)
        l_base = np.concatenate(l) if l else np.zeros(0, dtype=float)
        u_base = np.concatenate(u) if u else np.zeros(0, dtype=float)

        # Variable bounds via stacked identity constraints.
        l_var = np.full(nvar, -np.inf, dtype=float)
        u_var = np.full(nvar, np.inf, dtype=float)

        if (p.u_min is not None) and (p.u_max is not None):
            u_min = np.asarray(p.u_min, dtype=float).reshape(nu)
            u_max = np.asarray(p.u_max, dtype=float).reshape(nu)
            for k in range(N):
                uk = idx.u(k)
                l_var[uk] = u_min
                u_var[uk] = u_max

        if have_v:
            for k in range(N + 1):
                l_var[idx.sv(k)] = 0.0
        if have_a:
            for k in range(N + 1):
                l_var[idx.sa(k)] = 0.0

        # Stack [A_base; I]
        import scipy.sparse as sp
        import osqp

        A = np.vstack([A_base, np.eye(nvar, dtype=float)])
        l_full = np.concatenate([l_base, l_var])
        u_full = np.concatenate([u_base, u_var])

        P_sp = sp.csc_matrix((P + P.T) * 0.5)
        A_sp = sp.csc_matrix(A)

        solver = osqp.OSQP()
        solver.setup(
            P=P_sp,
            q=np.zeros(nvar),
            A=A_sp,
            l=l_full,
            u=u_full,
            verbose=False,
            warm_start=True,
        )

        self._idx = idx
        self._solver = solver
        self._P = P_sp
        self._A = A_sp
        self._l_base = l_base
        self._u_base = u_base
        self._l_var = l_var
        self._u_var = u_var
        self._rows_x0 = rows_x0
        self._rows_du0 = rows_du0
        self._rows_base = A_base.shape[0]

    def solve(self, x0, u0, target, x_ws, u_ws, log: bool = False):
        if self._solver is None or self._idx is None:
            raise RuntimeError("Call setup() before solve().")

        idx = self._idx
        p = self.params
        N = idx.N
        nx = idx.nx
        nu = idx.nu

        x0 = np.asarray(x0, dtype=float).reshape(nx)
        u0 = np.asarray(u0, dtype=float).reshape(nu)
        target = np.asarray(target, dtype=float).reshape(len(p.track_idx))

        # q term (tracking + du0)
        q = np.zeros(idx.nvar, dtype=float)
        track_idx = np.asarray(p.track_idx, dtype=int)
        Q = np.asarray(p.Q, dtype=float)

        for k in range(N):
            xk = idx.x(k)
            q[xk.start + track_idx] += -2.0 * (Q @ target)
        xN = idx.x(N)
        q[xN.start + track_idx] += -2.0 * float(p.terminal) * (Q @ target)

        Rd = np.asarray(p.Rd, dtype=float)
        q[idx.u(0)] += -2.0 * (Rd @ u0)

        # l/u (update x0 equality + du0 bounds)
        l_base = np.array(self._l_base, copy=True)
        u_base = np.array(self._u_base, copy=True)
        l_base[self._rows_x0] = x0
        u_base[self._rows_x0] = x0

        if (
            self._rows_du0 is not None
            and (p.du_min is not None)
            and (p.du_max is not None)
        ):
            du_min = np.asarray(p.du_min, dtype=float).reshape(nu)
            du_max = np.asarray(p.du_max, dtype=float).reshape(nu)
            l_base[self._rows_du0] = u0 + float(p.dt) * du_min
            u_base[self._rows_du0] = u0 + float(p.dt) * du_max

        l = np.concatenate([l_base, self._l_var])
        u = np.concatenate([u_base, self._u_var])

        # Warm start (shift by one step)
        x_ws = np.asarray(x_ws, dtype=float).reshape(nx, N + 1)
        u_ws = np.asarray(u_ws, dtype=float).reshape(nu, N)
        x_ws = np.hstack([x_ws[:, 1:], x_ws[:, -1:]])
        u_ws = np.hstack([u_ws[:, 1:], u_ws[:, -1:]]) if N > 1 else u_ws.copy()

        z0 = np.zeros(idx.nvar, dtype=float)
        for k in range(N + 1):
            z0[idx.x(k)] = x_ws[:, k]
        for k in range(N):
            z0[idx.u(k)] = u_ws[:, k]

        self._solver.update(q=q, l=l, u=u)
        self._solver.warm_start(x=z0)

        t0 = time.perf_counter()
        res = self._solver.solve()
        dt_s = time.perf_counter() - t0
        self.solver_times.append(dt_s)

        if log:
            print(
                f"solver time: {dt_s:.6f}s, "
                f"max: {max(self.solver_times):.6f}s, "
                f"mean: {float(np.mean(self.solver_times)):.6f}s, "
                f"status: {res.info.status}"
            )

        if res.x is None:
            raise RuntimeError(f"OSQP failed: {res.info.status}")

        z = np.asarray(res.x, dtype=float).reshape(-1)
        x_sol = np.zeros((nx, N + 1), dtype=float)
        u_sol = np.zeros((nu, N), dtype=float)
        for k in range(N + 1):
            x_sol[:, k] = z[idx.x(k)]
        for k in range(N):
            u_sol[:, k] = z[idx.u(k)]

        return x_sol, u_sol
