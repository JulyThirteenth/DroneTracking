import time
from dataclasses import dataclass
import numpy as np

from utils.utils import uav_jerk_discrete_matrices

try:
    import osqp  # noqa: F401
    import scipy.sparse as sp  # noqa: F401
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("Tracker QP requires `osqp` and `scipy`.") from exc


@dataclass(frozen=True)
class _Idx:
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


class TrackerOSQP:
    """
    QP (OSQP) implementation of `chaser.Tracker` MPCC.

    Interface matches `chaser.Tracker.solve()`:
      solve(x0, u0, target, x_ws, u_ws, s_ws=None, vs_ws=None) -> (x, u, s, vs)
    """

    def __init__(self, params):
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

        self._q_const: np.ndarray | None = None
        self._P_pos: dict[tuple[int, int], int] | None = None

    def setup(self):
        p = self.params
        nx = int(getattr(p, "nx"))
        nu = int(getattr(p, "nu"))
        N = int(getattr(p, "horizon"))
        dt = float(getattr(p, "dt"))

        if nx != 9 or nu != 3:
            raise ValueError("Tracker QP supports nx=9, nu=3 (UAV p,v,a + jerk).")

        have_v = (getattr(p, "v_min", None) is not None) and (
            getattr(p, "v_max", None) is not None
        )
        have_a = (getattr(p, "a_min", None) is not None) and (
            getattr(p, "a_max", None) is not None
        )

        x0 = 0
        u0 = nx * (N + 1)
        s0 = u0 + nu * N
        vs0 = s0 + (N + 1)
        sv0 = vs0 + N if have_v else None
        sa0 = (sv0 + 3 * (N + 1)) if have_v else (vs0 + N)
        sa0 = sa0 if have_a else None
        idx = _Idx(nx=nx, nu=nu, N=N, x0=x0, u0=u0, s0=s0, vs0=vs0, sv0=sv0, sa0=sa0)
        nvar = idx.nvar

        # --------------------
        # Objective: 0.5 z' P z + q' z
        # --------------------
        P = np.zeros((nvar, nvar), dtype=float)

        # u cost
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

        # MPCC: add constant s quadratic (lag) and terminal s quadratic (reach end)
        q_l = float(getattr(p, "q_lag", 2.0))
        q_term = float(getattr(p, "q_terminal_s", 50.0))
        for k in range(N):
            sk = idx.s(k)
            P[sk, sk] += 2.0 * q_l
        P[idx.s(N), idx.s(N)] += 2.0 * q_term

        # slack costs
        if have_v:
            w = float(getattr(p, "v_slack_weight", 50.0))
            for k in range(N + 1):
                sk = idx.sv(k)
                P[sk, sk] += 2.0 * w * np.eye(3)
        if have_a:
            w = float(getattr(p, "a_slack_weight", 50.0))
            for k in range(N + 1):
                sk = idx.sa(k)
                P[sk, sk] += 2.0 * w * np.eye(3)

        # Include a dense p block pattern and p-s cross pattern for k=0..N-1 (values updated at solve).
        # Use a non-axis-aligned direction to ensure all relevant entries exist in the sparse pattern.
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

        # constant linear term: -q_progress * vs
        q_const = np.zeros(nvar, dtype=float)
        q_s = float(getattr(p, "q_progress", 5.0))
        for k in range(N):
            q_const[idx.vs(k)] += -q_s

        # --------------------
        # Constraints: l <= A z <= u
        # --------------------
        A_d, B_d = uav_jerk_discrete_matrices(dt)

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

        # s0 == 0
        As0 = np.zeros((1, nvar), dtype=float)
        As0[0, idx.s(0)] = 1.0
        add(As0, np.zeros(1), np.zeros(1))

        # dynamics: x_{k+1} - A x_k - B u_k == 0
        for k in range(N):
            Ak = np.zeros((nx, nvar), dtype=float)
            Ak[:, idx.x(k + 1)] = np.eye(nx)
            Ak[:, idx.x(k)] = -A_d
            Ak[:, idx.u(k)] = -B_d
            add(Ak, np.zeros(nx), np.zeros(nx))

        # progress dynamics: s_{k+1} - s_k - dt*vs_k == 0
        for k in range(N):
            Ak = np.zeros((1, nvar), dtype=float)
            Ak[0, idx.s(k + 1)] = 1.0
            Ak[0, idx.s(k)] = -1.0
            Ak[0, idx.vs(k)] = -dt
            add(Ak, np.zeros(1), np.zeros(1))

        rows_du0 = None
        if (getattr(p, "du_min", None) is not None) and (
            getattr(p, "du_max", None) is not None
        ):
            du_min = np.asarray(p.du_min, dtype=float).reshape(nu)
            du_max = np.asarray(p.du_max, dtype=float).reshape(nu)
            lb_du = dt * du_min
            ub_du = dt * du_max

            for k in range(N - 1):
                Ak = np.zeros((nu, nvar), dtype=float)
                Ak[:, idx.u(k + 1)] = np.eye(nu)
                Ak[:, idx.u(k)] = -np.eye(nu)
                add(Ak, lb_du, ub_du)

            # u0 in [u_init+..., u_init+...]
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

                Ak = np.zeros((3, nvar), dtype=float)
                Ak[:, xk.start + 3 : xk.start + 6] = np.eye(3)
                Ak[:, idx.sv(k)] = np.eye(3)
                add(Ak, v_min, np.full(3, np.inf))

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

        if (getattr(p, "u_min", None) is not None) and (
            getattr(p, "u_max", None) is not None
        ):
            u_min = np.asarray(p.u_min, dtype=float).reshape(nu)
            u_max = np.asarray(p.u_max, dtype=float).reshape(nu)
            for k in range(N):
                uk = idx.u(k)
                l_var[uk] = u_min
                u_var[uk] = u_max

        # vs >= 0 and optional bounds
        vs_min = getattr(p, "vs_min", None)
        vs_max = getattr(p, "vs_max", None)
        lo_vs = 0.0 if vs_min is None else max(0.0, float(vs_min))
        hi_vs = np.inf if vs_max is None else float(vs_max)
        for k in range(N):
            vsk = idx.vs(k)
            l_var[vsk] = lo_vs
            u_var[vsk] = hi_vs

        if have_v:
            for k in range(N + 1):
                l_var[idx.sv(k)] = 0.0
        if have_a:
            for k in range(N + 1):
                l_var[idx.sa(k)] = 0.0

        A = np.vstack([A_base, np.eye(nvar, dtype=float)])
        l_full = np.concatenate([l_base, l_var])
        u_full = np.concatenate([u_base, u_var])

        # OSQP requires P in upper-triangular form; keep the sparsity pattern fixed for Px updates.
        P_sp = sp.triu(sp.csc_matrix((P + P.T) * 0.5), format="csc")
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

        # Build (row,col)->data_index mapping for fast P updates.
        P_pos: dict[tuple[int, int], int] = {}
        for col in range(P_sp.shape[1]):
            start = int(P_sp.indptr[col])
            end = int(P_sp.indptr[col + 1])
            for data_i in range(start, end):
                row = int(P_sp.indices[data_i])
                P_pos[(row, col)] = data_i

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
        self._q_const = q_const
        self._P_pos = P_pos

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
            self._solver is None
            or self._idx is None
            or self._P is None
            or self._P_pos is None
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
        target = np.asarray(target, dtype=float).reshape(-1)
        if target.shape[0] < 3:
            raise ValueError("target must contain at least 3 elements (goal position).")

        # Update P entries that depend on the reference line.
        q_c = float(getattr(p, "q_contour", 20.0))
        q_l = float(getattr(p, "q_lag", 2.0))

        p0 = x0[0:3]
        pg = target[0:3]
        d = pg - p0
        eps = 1e-6
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

        # Overwrite the (p,p) and (p,s)/(s,p) blocks for k=0..N-1.
        P_sp = self._P
        P_pos = self._P_pos
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

        # Build q: constant vs term + line-dependent linear terms + du0 term.
        q = np.array(self._q_const, copy=True)

        c = float(np.dot(t, p0))
        lin_p = (-2.0 * q_c) * (Pperp @ p0) + (-2.0 * q_l * c) * t
        lin_s = 2.0 * q_l * c
        for k in range(N):
            base = idx.x(k).start
            q[base + 0 : base + 3] += lin_p
            q[idx.s(k)] += lin_s

        q_term = float(getattr(p, "q_terminal_s", 50.0))
        q[idx.s(N)] += -2.0 * q_term * L

        Rd = (
            np.diag([0.2, 0.2, 0.2])
            if getattr(p, "Rd", None) is None
            else np.asarray(p.Rd, dtype=float)
        )
        q[idx.u(0)] += -2.0 * (Rd @ u0)

        # l/u (update x0 equality + du0 bounds)
        l_base = np.array(self._l_base, copy=True)
        u_base = np.array(self._u_base, copy=True)
        l_base[self._rows_x0] = x0
        u_base[self._rows_x0] = x0

        if (
            self._rows_du0 is not None
            and (getattr(p, "du_min", None) is not None)
            and (getattr(p, "du_max", None) is not None)
        ):
            du_min = np.asarray(p.du_min, dtype=float).reshape(nu)
            du_max = np.asarray(p.du_max, dtype=float).reshape(nu)
            l_base[self._rows_du0] = u0 + dt * du_min
            u_base[self._rows_du0] = u0 + dt * du_max

        l = np.concatenate([l_base, self._l_var])
        u = np.concatenate([u_base, self._u_var])

        # Warm start (shift by one step)
        x_ws = np.asarray(x_ws, dtype=float).reshape(nx, N + 1)
        u_ws = np.asarray(u_ws, dtype=float).reshape(nu, N)
        x_ws = np.hstack([x_ws[:, 1:], x_ws[:, -1:]])
        u_ws = np.hstack([u_ws[:, 1:], u_ws[:, -1:]]) if N > 1 else u_ws.copy()

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

        self._solver.update(Px=self._P.data, q=q, l=l, u=u)
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
        s_sol = np.zeros((1, N + 1), dtype=float)
        vs_sol = np.zeros((1, N), dtype=float)

        for k in range(N + 1):
            x_sol[:, k] = z[idx.x(k)]
            s_sol[0, k] = z[idx.s(k)]
        for k in range(N):
            u_sol[:, k] = z[idx.u(k)]
            vs_sol[0, k] = z[idx.vs(k)]

        return x_sol, u_sol, s_sol, vs_sol
