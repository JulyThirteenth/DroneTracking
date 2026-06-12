from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from tracking.tracking_utils import (
    Polyline3D,
    coerce_path_points_3d,
    sample_polyline_with_tangent,
)


@dataclass(frozen=True)
class HOCBFConfig:
    """High-order CBF settings for triple-integrator obstacle avoidance."""

    max_obstacles: int = 90
    safe_distance: float = 0.9
    lambda_gain: float = 0.8
    slack_weight: float = 1.0e6

    @property
    def gains(self) -> tuple[float, float, float]:
        lam = float(self.lambda_gain)
        return lam**3, 3.0 * lam * lam, 3.0 * lam


def uav_jerk_discrete_matrices(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Discrete triple-integrator for x=[p,v,a], u=jerk (all 3D), held constant on [k,k+1):
      p+ = p + dt v + 0.5 dt^2 a + (1/6) dt^3 u
      v+ = v + dt a + 0.5 dt^2 u
      a+ = a + dt u
    """
    dt = float(dt)
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))

    A = np.block(
        [
            [I3, dt * I3, 0.5 * dt * dt * I3],
            [Z3, I3, dt * I3],
            [Z3, Z3, I3],
        ]
    )
    B = np.block(
        [
            [(dt**3) / 6.0 * I3],
            [0.5 * dt * dt * I3],
            [dt * I3],
        ]
    )
    return A, B


class OSQPInterface:
    def __init__(self, name: str):
        self.name = str(name)
        self.solver = None
        self.solver_times: list[float] = []

    @property
    def ready(self) -> bool:
        return self.solver is not None

    @staticmethod
    def sparse_module(name: str):
        try:
            import scipy.sparse as sp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(f"{name} requires `scipy`.") from exc
        return sp

    def setup(self, *, P, q, A, l, u, **settings) -> None:
        try:
            import osqp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(f"{self.name} requires `osqp`.") from exc

        opts = {"verbose": False, "warm_start": True}
        opts.update(settings)
        self.solver = osqp.OSQP()
        self.solver.setup(P=P, q=q, A=A, l=l, u=u, **opts)

    def solve(
        self,
        *,
        q,
        l,
        u,
        warm_start=None,
        Px=None,
        Ax=None,
        log: bool = False,
    ) -> np.ndarray:
        if self.solver is None:
            raise RuntimeError("Call setup() before solve().")

        update = {"q": q, "l": l, "u": u}
        if Px is not None:
            update["Px"] = Px
        if Ax is not None:
            update["Ax"] = Ax
        self.solver.update(**update)
        if warm_start is not None:
            self.solver.warm_start(x=warm_start)

        t0 = time.perf_counter()
        res = self.solver.solve()
        dt_s = time.perf_counter() - t0
        self.solver_times.append(dt_s)

        if log:
            print(
                f"solver time: {dt_s:.6f}s, "
                f"max: {max(self.solver_times):.6f}s, "
                f"mean: {float(np.mean(self.solver_times)):.6f}s, "
                f"status: {res.info.status}"
            )

        status = str(res.info.status).lower()
        if status not in {"solved", "solved inaccurate"} or res.x is None:
            raise RuntimeError(f"OSQP failed: {res.info.status}")
        return np.asarray(res.x, dtype=float).reshape(-1)


class OSQPProblemBase:
    def __init__(self, name: str, params):
        self.params = params
        self._osqp = OSQPInterface(name)
        self._idx = None
        self._l_base: np.ndarray | None = None
        self._u_base: np.ndarray | None = None
        self._l_var: np.ndarray | None = None
        self._u_var: np.ndarray | None = None
        self._rows_x0: slice | None = None
        self._rows_du0: slice | None = None

    def _dims(self, name: str):
        p = self.params
        nx = int(getattr(p, "nx"))
        nu = int(getattr(p, "nu"))
        N = int(getattr(p, "horizon"))
        dt = float(getattr(p, "dt"))
        if nx != 9 or nu != 3:
            raise ValueError(f"{name} supports nx=9, nu=3 (UAV p,v,a + jerk).")
        return p, nx, nu, N, dt

    def _has_v_bounds(self) -> bool:
        p = self.params
        return getattr(p, "v_min", None) is not None and getattr(p, "v_max", None) is not None

    def _has_a_bounds(self) -> bool:
        p = self.params
        return getattr(p, "a_min", None) is not None and getattr(p, "a_max", None) is not None

    @staticmethod
    def _add_input_cost(P: np.ndarray, idx, N: int, R: np.ndarray, Rd: np.ndarray) -> None:
        for k in range(N):
            P[idx.u(k), idx.u(k)] += 2.0 * R
        for k in range(N - 1):
            uk = idx.u(k)
            uk1 = idx.u(k + 1)
            P[uk, uk] += 2.0 * Rd
            P[uk1, uk1] += 2.0 * Rd
            P[uk, uk1] += -2.0 * Rd
            P[uk1, uk] += -2.0 * Rd
        P[idx.u(0), idx.u(0)] += 2.0 * Rd

    def _add_soft_cost(self, P: np.ndarray, idx, N: int, *, have_v: bool, have_a: bool) -> None:
        p = self.params
        if have_v:
            w = float(getattr(p, "v_slack_weight"))
            for k in range(N + 1):
                P[idx.sv(k), idx.sv(k)] += 2.0 * w * np.eye(3)
        if have_a:
            w = float(getattr(p, "a_slack_weight"))
            for k in range(N + 1):
                P[idx.sa(k), idx.sa(k)] += 2.0 * w * np.eye(3)

    def _add_dynamics(self, rows, l, u, idx, *, nx: int, N: int, nvar: int, dt: float) -> None:
        A_d, B_d = uav_jerk_discrete_matrices(dt)
        for k in range(N):
            Ak = np.zeros((nx, nvar), dtype=float)
            Ak[:, idx.x(k + 1)] = np.eye(nx)
            Ak[:, idx.x(k)] = -A_d
            Ak[:, idx.u(k)] = -B_d
            rows.append(Ak)
            l.append(np.zeros(nx))
            u.append(np.zeros(nx))

    def _add_du_constraints(self, rows, l, u, idx, *, nu: int, N: int, nvar: int, dt: float) -> slice | None:
        p = self.params
        if getattr(p, "du_min", None) is None or getattr(p, "du_max", None) is None:
            return None
        du_min = np.asarray(getattr(p, "du_min"), dtype=float).reshape(nu)
        du_max = np.asarray(getattr(p, "du_max"), dtype=float).reshape(nu)
        lb_du = dt * du_min
        ub_du = dt * du_max

        for k in range(N - 1):
            Ak = np.zeros((nu, nvar), dtype=float)
            Ak[:, idx.u(k + 1)] = np.eye(nu)
            Ak[:, idx.u(k)] = -np.eye(nu)
            rows.append(Ak)
            l.append(lb_du)
            u.append(ub_du)

        Ak = np.zeros((nu, nvar), dtype=float)
        Ak[:, idx.u(0)] = np.eye(nu)
        start = sum(r.shape[0] for r in rows)
        rows.append(Ak)
        l.append(lb_du)
        u.append(ub_du)
        return slice(start, start + nu)

    def _add_soft_constraints(self, rows, l, u, idx, *, N: int, nvar: int, have_v: bool, have_a: bool) -> None:
        p = self.params
        if have_v:
            v_min = np.asarray(getattr(p, "v_min"), dtype=float).reshape(3)
            v_max = np.asarray(getattr(p, "v_max"), dtype=float).reshape(3)
            for k in range(N + 1):
                xk = idx.x(k)
                Ak = np.zeros((3, nvar), dtype=float)
                Ak[:, xk.start + 3 : xk.start + 6] = np.eye(3)
                Ak[:, idx.sv(k)] = np.eye(3)
                rows.append(Ak)
                l.append(v_min)
                u.append(np.full(3, np.inf))

                Ak = np.zeros((3, nvar), dtype=float)
                Ak[:, xk.start + 3 : xk.start + 6] = np.eye(3)
                Ak[:, idx.sv(k)] = -np.eye(3)
                rows.append(Ak)
                l.append(np.full(3, -np.inf))
                u.append(v_max)

        if have_a:
            a_min = np.asarray(getattr(p, "a_min"), dtype=float).reshape(3)
            a_max = np.asarray(getattr(p, "a_max"), dtype=float).reshape(3)
            for k in range(N + 1):
                xk = idx.x(k)
                Ak = np.zeros((3, nvar), dtype=float)
                Ak[:, xk.start + 6 : xk.start + 9] = np.eye(3)
                Ak[:, idx.sa(k)] = np.eye(3)
                rows.append(Ak)
                l.append(a_min)
                u.append(np.full(3, np.inf))

                Ak = np.zeros((3, nvar), dtype=float)
                Ak[:, xk.start + 6 : xk.start + 9] = np.eye(3)
                Ak[:, idx.sa(k)] = -np.eye(3)
                rows.append(Ak)
                l.append(np.full(3, -np.inf))
                u.append(a_max)

    def _var_bounds(self, idx, *, nvar: int, nu: int, N: int, have_v: bool, have_a: bool):
        p = self.params
        l_var = np.full(nvar, -np.inf, dtype=float)
        u_var = np.full(nvar, np.inf, dtype=float)
        if getattr(p, "u_min", None) is not None and getattr(p, "u_max", None) is not None:
            u_min = np.asarray(getattr(p, "u_min"), dtype=float).reshape(nu)
            u_max = np.asarray(getattr(p, "u_max"), dtype=float).reshape(nu)
            for k in range(N):
                l_var[idx.u(k)] = u_min
                u_var[idx.u(k)] = u_max
        if have_v:
            for k in range(N + 1):
                l_var[idx.sv(k)] = 0.0
        if have_a:
            for k in range(N + 1):
                l_var[idx.sa(k)] = 0.0
        return l_var, u_var

    @staticmethod
    def _stack_rows(rows, l, u, nvar: int):
        return (
            np.vstack(rows) if rows else np.zeros((0, nvar), dtype=float),
            np.concatenate(l) if l else np.zeros(0, dtype=float),
            np.concatenate(u) if u else np.zeros(0, dtype=float),
        )

    @staticmethod
    def _shift_xu_warmstart(x_ws, u_ws, *, nx: int, nu: int, N: int):
        x_ws = np.asarray(x_ws, dtype=float).reshape(nx, N + 1)
        u_ws = np.asarray(u_ws, dtype=float).reshape(nu, N)
        x_ws = np.hstack([x_ws[:, 1:], x_ws[:, -1:]])
        u_ws = np.hstack([u_ws[:, 1:], u_ws[:, -1:]]) if N > 1 else u_ws.copy()
        return x_ws, u_ws

    def _bounds_for_state_and_du(self, x0: np.ndarray, u0: np.ndarray, *, nu: int, dt: float):
        p = self.params
        l_base = np.array(self._l_base, copy=True)
        u_base = np.array(self._u_base, copy=True)
        l_base[self._rows_x0] = x0
        u_base[self._rows_x0] = x0

        if (
            self._rows_du0 is not None
            and getattr(p, "du_min", None) is not None
            and getattr(p, "du_max", None) is not None
        ):
            du_min = np.asarray(getattr(p, "du_min"), dtype=float).reshape(nu)
            du_max = np.asarray(getattr(p, "du_max"), dtype=float).reshape(nu)
            l_base[self._rows_du0] = u0 + dt * du_min
            u_base[self._rows_du0] = u0 + dt * du_max
        return l_base, u_base

    @staticmethod
    def _extract_xu(z, idx):
        x_sol = np.zeros((idx.nx, idx.N + 1), dtype=float)
        u_sol = np.zeros((idx.nu, idx.N), dtype=float)
        for k in range(idx.N + 1):
            x_sol[:, k] = z[idx.x(k)]
        for k in range(idx.N):
            u_sol[:, k] = z[idx.u(k)]
        return x_sol, u_sol


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


class MPCOSQP(OSQPProblemBase):
    """
    OSQP-based solver for the convex QP form of `tracking_opt.MPC`.

    Differences vs `qp_opt.ChaserOSQP`:
    - The tracked reference may be a per-step trajectory: `ref_traj` with shape (n_track, N+1).
    """

    def __init__(self, params, cbf=None):
        super().__init__("MPCOSQP", params)
        self.cbf = cbf
        self._rows_cbf: slice | None = None
        self._cbf_slack0: int | None = None
        self._nvar: int | None = None
        self._A = None
        self._A_pos: dict[tuple[int, int], int] = {}

    def setup(self):
        p, nx, nu, N, dt = self._dims("MPCOSQP")
        K = int(getattr(self.cbf, "max_obstacles", 0)) if self.cbf is not None else 0
        have_v = self._has_v_bounds()
        have_a = self._has_a_bounds()

        # 优化变量 z = [
        #   x_0, x_1, ..., x_N,
        #   u_0, u_1, ..., u_{N-1},
        #   sv_0, sv_1, ..., sv_N,
        #   sa_0, sa_1, ..., sa_N,
        #   cbf_slack_0, ..., cbf_slack_{N*K-1}
        # ]
        x0 = 0
        u0 = nx * (N + 1)
        sv0 = u0 + nu * N if have_v else None
        sa0 = (sv0 + 3 * (N + 1)) if have_v else (u0 + nu * N)
        sa0 = sa0 if have_a else None
        idx = _Idx(nx=nx, nu=nu, N=N, x0=x0, u0=u0, sv0=sv0, sa0=sa0)
        cbf_slack0 = idx.nvar if K > 0 else None
        nvar = idx.nvar + (N * K if K > 0 else 0) 

        # --------------------
        # Objective: 0.5 z' P z + q' z
        # --------------------
        P = np.zeros((nvar, nvar), dtype=float)
        
        # [px_i, py_i, pz_i]_{N+1}^T @ Q @ [px_i, py_i, pz_i]_{N+1}
        track_idx = np.asarray(getattr(p, "track_idx"), dtype=int)
        Q = np.asarray(getattr(p, "Q"), dtype=float)
        for k in range(N):
            xk = idx.x(k)
            P[np.ix_(xk.start + track_idx, xk.start + track_idx)] += 2.0 * Q
        xN = idx.x(N)
        P[np.ix_(xN.start + track_idx, xN.start + track_idx)] += (
            2.0 * float(getattr(p, "terminal")) * Q
        )
        # [jx_i, jy_i, jz_i]_{N}^T @ R @ [jx_i, jy_i, jz_i]_{N}
        R = np.asarray(getattr(p, "R"), dtype=float)
        Rd = np.asarray(getattr(p, "Rd"), dtype=float)
        self._add_input_cost(P, idx, N, R, Rd)
        self._add_soft_cost(P, idx, N, have_v=have_v, have_a=have_a)
        if cbf_slack0 is not None:
            w = float(getattr(self.cbf, "slack_weight"))
            for i in range(N * K):
                P[cbf_slack0 + i, cbf_slack0 + i] += 2.0 * w

        # --------------------
        # Constraints: l <= A z <= u
        # --------------------
        rows: list[np.ndarray] = []
        l: list[np.ndarray] = []
        u: list[np.ndarray] = []

        # x0 == x_init (filled at solve via bounds)
        A0 = np.zeros((nx, nvar), dtype=float)
        A0[:, idx.x(0)] = np.eye(nx)
        rows.append(A0)
        l.append(np.zeros(nx))
        u.append(np.zeros(nx))
        rows_x0 = slice(0, nx)

        # dynamics: x_{k+1} - A x_k - B u_k == 0
        self._add_dynamics(rows, l, u, idx, nx=nx, N=N, nvar=nvar, dt=dt)
        rows_du0 = self._add_du_constraints(rows, l, u, idx, nu=nu, N=N, nvar=nvar, dt=dt)
        self._add_soft_constraints(rows, l, u, idx, N=N, nvar=nvar, have_v=have_v, have_a=have_a)

        A_base, l_base, u_base = self._stack_rows(rows, l, u, nvar)

        if K > 0:
            cbf_start = A_base.shape[0]
            cbf_rows = np.zeros((N * K, nvar), dtype=float)
            eps = 1e-12
            for k in range(N):
                xk = idx.x(k)
                uk = idx.u(k)
                cols = list(range(xk.start, xk.start + 9)) + list(
                    range(uk.start, uk.start + 3)
                )
                for i in range(K):
                    cbf_rows[k * K + i, cols + [cbf_slack0 + k * K + i]] = eps
            A_base = np.vstack([A_base, cbf_rows])
            l_base = np.concatenate([l_base, np.full(N * K, -np.inf)])
            u_base = np.concatenate([u_base, np.full(N * K, np.inf)])
            rows_cbf = slice(cbf_start, cbf_start + N * K)
        else:
            rows_cbf = slice(A_base.shape[0], A_base.shape[0])

        l_var, u_var = self._var_bounds(
            idx, nvar=nvar, nu=nu, N=N, have_v=have_v, have_a=have_a
        )
        if cbf_slack0 is not None:
            l_var[cbf_slack0 : cbf_slack0 + N * K] = 0.0

        # Stack [A_base; I]
        sp = OSQPInterface.sparse_module("MPCOSQP")

        A = np.vstack([A_base, np.eye(nvar, dtype=float)])
        l_full = np.concatenate([l_base, l_var])
        u_full = np.concatenate([u_base, u_var])

        P_sp = sp.csc_matrix((P + P.T) * 0.5)
        A_sp = sp.csc_matrix(A)

        self._osqp.setup(
            P=P_sp,
            q=np.zeros(nvar),
            A=A_sp,
            l=l_full,
            u=u_full,
            verbose=False,
            warm_start=True,
            max_iter=10000,
        )

        A_pos: dict[tuple[int, int], int] = {}
        if K > 0:
            for col in range(A_sp.shape[1]):
                start = int(A_sp.indptr[col])
                end = int(A_sp.indptr[col + 1])
                for data_i in range(start, end):
                    A_pos[(int(A_sp.indices[data_i]), col)] = data_i

        self._idx = idx
        self._l_base = l_base
        self._u_base = u_base
        self._l_var = l_var
        self._u_var = u_var
        self._rows_x0 = rows_x0
        self._rows_du0 = rows_du0
        self._rows_cbf = rows_cbf
        self._cbf_slack0 = cbf_slack0
        self._nvar = nvar
        self._A = A_sp
        self._A_pos = A_pos

    @staticmethod
    def _coerce_ref_traj(ref, *, n_track: int, horizon: int) -> np.ndarray:
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

    def _set_cbf_rows(
        self,
        l_base: np.ndarray,
        u_base: np.ndarray,
        normals,
        points,
    ) -> None:
        if self.cbf is None:
            return
        if self._idx is None or self._A is None or self._rows_cbf is None:
            raise RuntimeError("Call setup() before solve().")

        idx = self._idx
        K = int(getattr(self.cbf, "max_obstacles"))
        normals = np.asarray(normals, dtype=float).reshape(-1, 3)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        count = min(K, normals.shape[0], points.shape[0])
        k0, k1, k2 = self.cbf.gains
        safe = float(getattr(self.cbf, "safe_distance"))

        for k in range(idx.N):
            xk = idx.x(k)
            uk = idx.u(k)
            cols = list(range(xk.start, xk.start + 9)) + list(
                range(uk.start, uk.start + 3)
            )
            for i in range(K):
                row = self._rows_cbf.start + k * K + i
                active = (
                    i < count
                    and np.isfinite(normals[i]).all()
                    and np.isfinite(points[i]).all()
                )
                n = normals[i] if active else np.zeros(3, dtype=float)
                obs = points[i] if active else np.zeros(3, dtype=float)
                coeffs = np.r_[k0 * n, k1 * n, k2 * n, n]

                for col, value in zip(cols, coeffs):
                    pos = self._A_pos.get((row, col))
                    if pos is not None:
                        self._A.data[pos] = float(value)
                if self._cbf_slack0 is not None:
                    slack_col = self._cbf_slack0 + k * K + i
                    pos = self._A_pos.get((row, slack_col))
                    if pos is not None:
                        self._A.data[pos] = 1.0 if active else 0.0

                l_base[row] = k0 * (float(np.dot(n, obs)) + safe) if active else -np.inf
                u_base[row] = np.inf

    def solve(
        self,
        x0,
        u0,
        ref,
        x_ws,
        u_ws,
        *,
        obstacle_normals=None,
        obstacle_points=None,
        log: bool = False,
    ):
        if not self._osqp.ready or self._idx is None:
            raise RuntimeError("Call setup() before solve().")

        idx = self._idx
        p = self.params
        N = idx.N
        nx = idx.nx
        nu = idx.nu

        x0 = np.asarray(x0, dtype=float).reshape(nx)
        u0 = np.asarray(u0, dtype=float).reshape(nu)

        track_idx = np.asarray(getattr(p, "track_idx"), dtype=int)
        Q = np.asarray(getattr(p, "Q"), dtype=float)
        Rd = np.asarray(getattr(p, "Rd"), dtype=float)
        terminal_w = float(getattr(p, "terminal"))

        ref_traj = self._coerce_ref_traj(ref, n_track=len(track_idx), horizon=N)

        # q term (tracking + du0)
        nvar = int(self._nvar or idx.nvar)
        q = np.zeros(nvar, dtype=float)
        for k in range(N):
            xk = idx.x(k)
            q[xk.start + track_idx] += -2.0 * (Q @ ref_traj[:, k])
        xN = idx.x(N)
        q[xN.start + track_idx] += -2.0 * terminal_w * (Q @ ref_traj[:, N])

        q[idx.u(0)] += -2.0 * (Rd @ u0)

        l_base, u_base = self._bounds_for_state_and_du(
            x0, u0, nu=nu, dt=float(getattr(p, "dt"))
        )

        if self.cbf is not None:
            if obstacle_normals is None or obstacle_points is None:
                obstacle_normals = np.zeros((0, 3), dtype=float)
                obstacle_points = np.zeros((0, 3), dtype=float)
            self._set_cbf_rows(l_base, u_base, obstacle_normals, obstacle_points)

        l = np.concatenate([l_base, self._l_var])
        u = np.concatenate([u_base, self._u_var])

        # Warm start (shift by one step)
        x_ws, u_ws = self._shift_xu_warmstart(x_ws, u_ws, nx=nx, nu=nu, N=N)

        z0 = np.zeros(nvar, dtype=float)
        for k in range(N + 1):
            z0[idx.x(k)] = x_ws[:, k]
        for k in range(N):
            z0[idx.u(k)] = u_ws[:, k]

        z = self._osqp.solve(
            q=q,
            l=l,
            u=u,
            Ax=None if self.cbf is None else self._A.data,
            warm_start=z0,
            log=log,
        )
        return self._extract_xu(z, idx)


@dataclass(frozen=True)
class _IdxTracker:
    nx: int
    nu: int
    N: int
    x0: int
    u0: int
    s0: int
    vs0: int
    sv0: int | None
    sa0: int | None

    @property
    def nvar(self) -> int:
        n = self.vs0 + self.N  # vs: N
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

    def s(self, k: int) -> int:
        return self.s0 + k

    def vs(self, k: int) -> int:
        return self.vs0 + k

    def sv(self, k: int) -> slice:
        assert self.sv0 is not None
        s = self.sv0 + 3 * k
        return slice(s, s + 3)

    def sa(self, k: int) -> slice:
        assert self.sa0 is not None
        s = self.sa0 + 3 * k
        return slice(s, s + 3)


class MPCCOSQP(OSQPProblemBase):
    """
    OSQP-based QP MPCC that mirrors `tracking_opt.MPCC`.

    Supports:
      solve(..., target=goal_xyz): straight-line MPCC (same as `examples/pre/mpcc.py`)
      solve(..., target=path_points): polyline path MPCC using per-step local tangents.
    """

    def __init__(self, params):
        super().__init__("MPCCOSQP", params)
        self._P = None
        self._P_pos: dict[tuple[int, int], int] | None = None

        self._rows_s0: slice | None = None

        self._q_const: np.ndarray | None = None

        self._path: Polyline3D | None = None
        self._path_points: np.ndarray | None = None  # (M,3)
        self._path_s: np.ndarray | None = None  # (M,)
        self._path_L: float | None = None

    def set_path(self, path_points):
        pts = coerce_path_points_3d(path_points)
        if pts is None:
            raise ValueError("set_path expects a 2D path array (M,3)/(3,M)")
        poly = Polyline3D.from_points(pts, dedupe=True)
        self._path = poly
        self._path_points = poly.points
        self._path_s = poly.s
        self._path_L = poly.length

    def setup(self):
        p, nx, nu, N, dt = self._dims("MPCCOSQP")
        have_v = self._has_v_bounds()
        have_a = self._has_a_bounds()

        x0 = 0
        u0 = nx * (N + 1)
        s0 = u0 + nu * N
        vs0 = s0 + (N + 1)
        sv0 = vs0 + N if have_v else None
        sa0 = (sv0 + 3 * (N + 1)) if have_v else (vs0 + N)
        sa0 = sa0 if have_a else None
        idx = _IdxTracker(
            nx=nx, nu=nu, N=N, x0=x0, u0=u0, s0=s0, vs0=vs0, sv0=sv0, sa0=sa0
        )
        nvar = idx.nvar

        # --------------------
        # Objective: 0.5 z' P z + q' z
        # --------------------
        P = np.zeros((nvar, nvar), dtype=float)

        R = (
            np.diag([0.1, 0.1, 0.1])
            if getattr(p, "R", None) is None
            else np.asarray(p.R, dtype=float)
        )
        Rd = (
            np.diag([0.2, 0.2, 0.2])
            if getattr(p, "Rd", None) is None
            else np.asarray(p.Rd, dtype=float)
        )

        self._add_input_cost(P, idx, N, R, Rd)

        q_l = float(getattr(p, "q_lag", 2.0))
        q_term = float(getattr(p, "q_terminal_s", 50.0))
        for k in range(N):
            P[idx.s(k), idx.s(k)] += 2.0 * q_l
        P[idx.s(N), idx.s(N)] += 2.0 * q_term

        self._add_soft_cost(P, idx, N, have_v=have_v, have_a=have_a)

        # Ensure sparsity pattern for per-step MPCC updates (p block + p-s cross).
        t0_nom = np.array([1.0, 1.0, 1.0], dtype=float)
        t0_nom = t0_nom / float(np.linalg.norm(t0_nom))
        Pperp_nom = np.eye(3) - np.outer(t0_nom, t0_nom)
        q_c = float(getattr(p, "q_contour", 20.0))
        M_nom = q_c * Pperp_nom + q_l * np.outer(t0_nom, t0_nom)
        Ppp_nom = 2.0 * M_nom
        ps_nom = -2.0 * q_l * t0_nom
        for k in range(N):
            base = idx.x(k).start
            p_idx = [base + 0, base + 1, base + 2]
            sk = idx.s(k)
            for ii in range(3):
                for jj in range(3):
                    P[p_idx[ii], p_idx[jj]] += Ppp_nom[ii, jj]
                P[p_idx[ii], sk] += ps_nom[ii]
                P[sk, p_idx[ii]] += ps_nom[ii]

        q_const = np.zeros(nvar, dtype=float)
        q_s = float(getattr(p, "q_progress", 5.0))
        for k in range(N):
            q_const[idx.vs(k)] += -q_s

        # --------------------
        # Constraints: l <= A z <= u
        # --------------------
        rows: list[np.ndarray] = []
        l: list[np.ndarray] = []
        u: list[np.ndarray] = []

        A0 = np.zeros((nx, nvar), dtype=float)
        A0[:, idx.x(0)] = np.eye(nx)
        rows.append(A0)
        l.append(np.zeros(nx))
        u.append(np.zeros(nx))
        rows_x0 = slice(0, nx)

        # s0 equality (value filled at solve)
        As0 = np.zeros((1, nvar), dtype=float)
        As0[0, idx.s(0)] = 1.0
        start_s0 = sum(r.shape[0] for r in rows)
        rows.append(As0)
        l.append(np.zeros(1))
        u.append(np.zeros(1))
        rows_s0 = slice(start_s0, start_s0 + 1)

        self._add_dynamics(rows, l, u, idx, nx=nx, N=N, nvar=nvar, dt=dt)

        for k in range(N):
            Ak = np.zeros((1, nvar), dtype=float)
            Ak[0, idx.s(k + 1)] = 1.0
            Ak[0, idx.s(k)] = -1.0
            Ak[0, idx.vs(k)] = -dt
            rows.append(Ak)
            l.append(np.zeros(1))
            u.append(np.zeros(1))

        rows_du0 = self._add_du_constraints(rows, l, u, idx, nu=nu, N=N, nvar=nvar, dt=dt)
        self._add_soft_constraints(rows, l, u, idx, N=N, nvar=nvar, have_v=have_v, have_a=have_a)

        A_base, l_base, u_base = self._stack_rows(rows, l, u, nvar)
        l_var, u_var = self._var_bounds(
            idx, nvar=nvar, nu=nu, N=N, have_v=have_v, have_a=have_a
        )

        # s >= 0 by default (upper bound filled at solve if path is known)
        for k in range(N + 1):
            l_var[idx.s(k)] = 0.0

        vs_min = getattr(p, "vs_min", None)
        vs_max = getattr(p, "vs_max", None)
        lo_vs = 0.0 if vs_min is None else max(0.0, float(vs_min))
        hi_vs = np.inf if vs_max is None else float(vs_max)
        for k in range(N):
            vsk = idx.vs(k)
            l_var[vsk] = lo_vs
            u_var[vsk] = hi_vs

        sp = OSQPInterface.sparse_module("MPCCOSQP")

        A = np.vstack([A_base, np.eye(nvar, dtype=float)])
        l_full = np.concatenate([l_base, l_var])
        u_full = np.concatenate([u_base, u_var])

        P_sp = sp.triu(sp.csc_matrix((P + P.T) * 0.5), format="csc")
        A_sp = sp.csc_matrix(A)

        self._osqp.setup(
            P=P_sp,
            q=np.zeros(nvar),
            A=A_sp,
            l=l_full,
            u=u_full,
            verbose=False,
            warm_start=True,
        )

        P_pos: dict[tuple[int, int], int] = {}
        for col in range(P_sp.shape[1]):
            start = int(P_sp.indptr[col])
            end = int(P_sp.indptr[col + 1])
            for data_i in range(start, end):
                row = int(P_sp.indices[data_i])
                P_pos[(row, col)] = data_i

        self._idx = idx
        self._P = P_sp
        self._P_pos = P_pos
        self._l_base = l_base
        self._u_base = u_base
        self._l_var = l_var
        self._u_var = u_var
        self._rows_x0 = rows_x0
        self._rows_s0 = rows_s0
        self._rows_du0 = rows_du0
        self._q_const = q_const

    def _update_for_straight_line(self, x0: np.ndarray, target: np.ndarray):
        p = self.params
        idx = self._idx
        P_sp = self._P
        P_pos = self._P_pos
        assert idx is not None and P_sp is not None and P_pos is not None

        N = idx.N
        q_c = float(getattr(p, "q_contour", 20.0))
        q_l = float(getattr(p, "q_lag", 2.0))
        eps = 1e-6

        p0 = x0[0:3]
        pg = target[0:3]
        d = pg - p0
        L = float(np.linalg.norm(d))
        if L < eps:
            t = np.array([1.0, 0.0, 0.0], dtype=float)
            L = eps
        else:
            t = d / L

        Pperp = np.eye(3) - np.outer(t, t)
        M = q_c * Pperp + q_l * np.outer(t, t)
        Ppp = 2.0 * M
        ps = -2.0 * q_l * t

        for k in range(N):
            base = idx.x(k).start
            p_idx = [base + 0, base + 1, base + 2]
            sk = idx.s(k)
            for ii in range(3):
                for jj in range(3):
                    pos = P_pos.get((p_idx[ii], p_idx[jj]))
                    if pos is not None:
                        P_sp.data[pos] = Ppp[ii, jj]
                pos = P_pos.get((p_idx[ii], sk))
                if pos is not None:
                    P_sp.data[pos] = ps[ii]
                pos = P_pos.get((sk, p_idx[ii]))
                if pos is not None:
                    P_sp.data[pos] = ps[ii]

        q = np.array(self._q_const, copy=True)  # type: ignore[arg-type]
        c = float(np.dot(t, p0))
        lin_p = (-2.0 * q_c) * (Pperp @ p0) + (-2.0 * q_l * c) * t
        lin_s = 2.0 * q_l * c
        for k in range(N):
            base = idx.x(k).start
            q[base + 0 : base + 3] += lin_p
            q[idx.s(k)] += lin_s

        q_term = float(getattr(p, "q_terminal_s", 50.0))
        q[idx.s(N)] += -2.0 * q_term * L

        return q, float(L), float(0.0)

    def _update_for_path(self, x0: np.ndarray, path_pts: np.ndarray, s_bar: np.ndarray):
        p = self.params
        idx = self._idx
        P_sp = self._P
        P_pos = self._P_pos
        assert idx is not None and P_sp is not None and P_pos is not None

        N = idx.N
        q_c = float(getattr(p, "q_contour", 20.0))
        q_l = float(getattr(p, "q_lag", 2.0))
        eps = 1e-6

        # Ensure internal path cache matches input.
        if (
            self._path_points is None
            or self._path_s is None
            or self._path_L is None
            or self._path_points.shape != path_pts.shape
            or not np.allclose(self._path_points, path_pts, atol=1e-9, rtol=0.0)
        ):
            self.set_path(path_pts)

        pts = self._path_points
        cum = self._path_s
        L = float(self._path_L)

        q = np.array(self._q_const, copy=True)  # type: ignore[arg-type]

        for k in range(N):
            sb = max(0.0, min(float(s_bar[k]), L))
            pr, t = sample_polyline_with_tangent(pts, cum, sb)
            Pperp = np.eye(3) - np.outer(t, t)
            M = q_c * Pperp + q_l * np.outer(t, t)
            Ppp = 2.0 * M
            ps = -2.0 * q_l * t

            p0 = pr - t * sb
            c = float(np.dot(t, p0))

            lin_p = (-2.0 * q_c) * (Pperp @ p0) + (-2.0 * q_l * c) * t
            lin_s = 2.0 * q_l * c

            base = idx.x(k).start
            p_idx = [base + 0, base + 1, base + 2]
            sk = idx.s(k)

            for ii in range(3):
                for jj in range(3):
                    pos = P_pos.get((p_idx[ii], p_idx[jj]))
                    if pos is not None:
                        P_sp.data[pos] = Ppp[ii, jj]
                pos = P_pos.get((p_idx[ii], sk))
                if pos is not None:
                    P_sp.data[pos] = ps[ii]
                pos = P_pos.get((sk, p_idx[ii]))
                if pos is not None:
                    P_sp.data[pos] = ps[ii]

            q[base + 0 : base + 3] += lin_p
            q[sk] += lin_s

        q_term = float(getattr(p, "q_terminal_s", 50.0))
        q[idx.s(N)] += -2.0 * q_term * L

        if self._path is None:
            raise RuntimeError("Path not initialized; call set_path() first.")
        s_init = self._path.closest_s(x0[0:3])
        return q, L, float(s_init)

    def solve(
        self,
        x0,
        u0,
        target,
        x_ws,
        u_ws,
        s_ws=None,
        vs_ws=None,
        log: bool = False,
    ):
        if (
            not self._osqp.ready
            or self._idx is None
            or self._P is None
            or self._P_pos is None
            or self._l_base is None
            or self._u_base is None
            or self._l_var is None
            or self._u_var is None
            or self._q_const is None
            or self._rows_x0 is None
            or self._rows_s0 is None
        ):
            raise RuntimeError("Call setup() before solve().")

        idx = self._idx
        p = self.params
        N = idx.N
        nx = idx.nx
        nu = idx.nu
        dt = float(getattr(p, "dt"))

        x0 = np.asarray(x0, dtype=float).reshape(nx)
        u0 = np.asarray(u0, dtype=float).reshape(nu)

        path_pts = coerce_path_points_3d(target)
        if path_pts is None:
            tgt = np.asarray(target, dtype=float).reshape(-1)
            if tgt.shape[0] < 3:
                raise ValueError(
                    "target must contain at least 3 elements (goal position)."
                )
            q, L, s_init = self._update_for_straight_line(x0, tgt)
            s_upper = float(L)
        else:
            poly = Polyline3D.from_points(path_pts, dedupe=True)
            path_pts = poly.points

            # nominal s trajectory for local linearization (shifted warmstart if provided)
            if s_ws is None:
                vs_guess = float(getattr(p, "vs_max", 2.0) or 2.0)
                vs_guess = max(0.5, min(vs_guess, 4.0))
                s_init_guess = poly.closest_s(x0[0:3])
                s_bar = s_init_guess + dt * vs_guess * np.arange(N + 1, dtype=float)
            else:
                s_ws_arr = np.asarray(s_ws, dtype=float).reshape(1, N + 1)
                s_bar = np.hstack([s_ws_arr[:, 1:], s_ws_arr[:, -1:]])[0]

            q, L, s_init = self._update_for_path(x0, path_pts, s_bar)
            s_upper = float(L)

        Rd = (
            np.diag([0.2, 0.2, 0.2])
            if getattr(p, "Rd", None) is None
            else np.asarray(p.Rd, dtype=float)
        )
        q[idx.u(0)] += -2.0 * (Rd @ u0)

        l_base, u_base = self._bounds_for_state_and_du(x0, u0, nu=nu, dt=dt)
        l_base[self._rows_s0] = np.array([s_init], dtype=float)
        u_base[self._rows_s0] = np.array([s_init], dtype=float)

        l_var = np.array(self._l_var, copy=True)
        u_var = np.array(self._u_var, copy=True)
        for k in range(N + 1):
            u_var[idx.s(k)] = s_upper

        l = np.concatenate([l_base, l_var])
        u = np.concatenate([u_base, u_var])

        x_ws, u_ws = self._shift_xu_warmstart(x_ws, u_ws, nx=nx, nu=nu, N=N)

        if s_ws is None:
            s_ws_arr = np.zeros((1, N + 1), dtype=float)
        else:
            s_ws_arr = np.asarray(s_ws, dtype=float).reshape(1, N + 1)
            s_ws_arr = np.hstack([s_ws_arr[:, 1:], s_ws_arr[:, -1:]])

        if vs_ws is None:
            vs_ws_arr = np.zeros((1, N), dtype=float)
        else:
            vs_ws_arr = np.asarray(vs_ws, dtype=float).reshape(1, N)
            vs_ws_arr = (
                np.hstack([vs_ws_arr[:, 1:], vs_ws_arr[:, -1:]])
                if N > 1
                else vs_ws_arr.copy()
            )

        z0 = np.zeros(idx.nvar, dtype=float)
        for k in range(N + 1):
            z0[idx.x(k)] = x_ws[:, k]
            z0[idx.s(k)] = s_ws_arr[0, k]
        for k in range(N):
            z0[idx.u(k)] = u_ws[:, k]
            z0[idx.vs(k)] = vs_ws_arr[0, k]

        z = self._osqp.solve(
            q=q,
            l=l,
            u=u,
            Px=self._P.data,
            warm_start=z0,
            log=log,
        )

        x_sol, u_sol = self._extract_xu(z, idx)
        s_sol = np.zeros((1, N + 1), dtype=float)
        vs_sol = np.zeros((1, N), dtype=float)
        for k in range(N + 1):
            s_sol[0, k] = z[idx.s(k)]
        for k in range(N):
            vs_sol[0, k] = z[idx.vs(k)]

        return x_sol, u_sol, s_sol, vs_sol
