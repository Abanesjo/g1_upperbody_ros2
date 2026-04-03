"""CBF-QP solver using OSQP.

Solves:  min ||dq - dq_ref||^2
         s.t. A_cbf @ dq >= b_cbf  (CBF constraints)
              dq_min <= dq <= dq_max  (velocity limits)

Uses a slack variable with large penalty for feasibility.
Reuses the OSQP instance across calls for efficiency.
"""

import numpy as np
from scipy import sparse

try:
    import osqp
    _HAS_OSQP = True
except ImportError:
    _HAS_OSQP = False

# Tiny value to keep zeros in CSC sparsity pattern.
# OSQP update(Ax=) requires the same nnz as setup.
_EPS = 1e-20


class CBFQPSolver:
    """QP solver for CBF safety filter."""

    def __init__(
        self,
        n_joints: int = 10,
        n_cbf: int = 3,
        slack_weight: float = 1e4,
    ):
        self.n = n_joints
        self.n_cbf = n_cbf
        self.slack_weight = slack_weight
        self._solver = None
        self._A_template = None  # fixed sparsity CSC

    def solve(
        self,
        dq_ref: np.ndarray,
        cbf_constraints: list,
        dq_min: np.ndarray,
        dq_max: np.ndarray,
    ) -> np.ndarray:
        """Solve the CBF-QP.

        Parameters
        ----------
        dq_ref : (n,) reference joint velocity
        cbf_constraints : list of (A_row, b_val) tuples
            Each constraint: A_row @ dq >= b_val
        dq_min, dq_max : (n,) joint velocity bounds

        Returns
        -------
        dq_safe : (n,) safe joint velocity
        """
        if _HAS_OSQP:
            return self._solve_osqp(
                dq_ref, cbf_constraints,
                dq_min, dq_max,
            )
        return self._solve_scipy(
            dq_ref, cbf_constraints,
            dq_min, dq_max,
        )

    def _build_A(self, cbf_constraints, dq_min, dq_max):
        """Build constraint matrix with fixed sparsity.

        Every entry is filled (eps for structural zeros)
        so CSC nnz never changes between calls.
        """
        n = self.n
        N = n + 1  # [dq, slack]
        n_cbf = len(cbf_constraints)
        n_rows = n_cbf + n + 1

        # Fill with eps so no true zeros in CSC
        A = np.full((n_rows, N), _EPS)
        l = np.empty(n_rows)
        u = np.empty(n_rows)

        # CBF: [A_row, -1] @ [dq, s] >= b
        for i, (A_row, b_val) in enumerate(cbf_constraints):
            A[i, :n] = np.where(
                np.abs(A_row) > _EPS, A_row, _EPS,
            )
            A[i, n] = -1.0
            l[i] = b_val
            u[i] = np.inf

        # Velocity bounds: dq_min <= dq_i <= dq_max
        off = n_cbf
        for i in range(n):
            A[off + i, i] = 1.0
            l[off + i] = dq_min[i]
            u[off + i] = dq_max[i]

        # Slack >= 0
        A[-1, n] = 1.0
        l[-1] = 0.0
        u[-1] = np.inf

        return sparse.csc_matrix(A), l, u

    def _solve_osqp(
        self, dq_ref, cbf_constraints, dq_min, dq_max,
    ):
        n = self.n
        N = n + 1
        n_cbf = len(cbf_constraints)

        # Linear cost
        q_vec = np.zeros(N)
        q_vec[:n] = -2.0 * dq_ref

        # Constraint matrix (fixed nnz)
        A_csc, l, u = self._build_A(
            cbf_constraints, dq_min, dq_max,
        )

        if self._solver is not None and n_cbf == self.n_cbf:
            # Reuse: update values only (same nnz)
            self._solver.update(
                q=q_vec,
                Ax=A_csc.data,
                l=l,
                u=u,
            )
        else:
            # First call or constraint count changed
            self.n_cbf = n_cbf
            P_diag = np.ones(N)
            P_diag[n:] = self.slack_weight
            P = sparse.diags(P_diag, format='csc') * 2.0

            self._solver = osqp.OSQP()
            self._solver.setup(
                P, q_vec, A_csc, l, u,
                verbose=False,
                eps_abs=1e-6,
                eps_rel=1e-6,
                max_iter=4000,
                warm_start=True,
            )

        result = self._solver.solve()
        if result.info.status in (
            'solved', 'solved_inaccurate',
        ):
            return result.x[:n]
        print(f'[QP FALLBACK] status={result.info.status}, n_cbf={n_cbf}')
        return np.clip(dq_ref, dq_min, dq_max)

    def _solve_scipy(
        self, dq_ref, cbf_constraints, dq_min, dq_max,
    ):
        """Fallback using scipy SLSQP."""
        from scipy.optimize import minimize

        n = self.n

        constraints = []
        for A_row, b_val in cbf_constraints:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, A=A_row, b=b_val: (
                    A @ x - b
                ),
                'jac': lambda x, A=A_row: A,
            })

        bounds = list(zip(dq_min, dq_max))

        res = minimize(
            lambda x: np.sum((x - dq_ref) ** 2),
            x0=np.clip(dq_ref, dq_min, dq_max),
            jac=lambda x: 2.0 * (x - dq_ref),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10},
        )

        if res.success:
            return res.x
        return np.clip(dq_ref, dq_min, dq_max)
