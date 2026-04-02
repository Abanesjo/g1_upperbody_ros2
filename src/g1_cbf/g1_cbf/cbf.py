"""CBF using dpax for differentiable collision proximity.

Supports two geometry modes:
- capsules: dpax.endpoints.proximity (phi = d_cl^2 - r_sum^2)
- boxes: dpax.polytopes.polytope_proximity (alpha = scaling factor)
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
from dpax.endpoints import proximity
from dpax.qp_utils import get_cost_terms, active_set_qp
from dpax.polytopes import polytope_proximity, polytope_proximity_grads


# Box halfspace normals (constant, 6 faces)
_BOX_A = jnp.array([
    [1, 0, 0.], [0, 1, 0.], [0, 0, 1.],
    [-1, 0, 0.], [0, -1, 0.], [0, 0, -1.],
])


def _box_b_from_body(body):
    """Smallest enclosing box half-extents for a capsule body.

    In body frame: X,Y = radius, Z = half_length.
    Returns b vector for Ax<=b halfspace form.
    """
    r = body['radius']
    h = body['half_length']
    extents = jnp.array([r, r, h, r, r, h])
    return extents


def _box_b_from_half_extents(half_extents):
    """b vector for Ax<=b from (3,) half-extents [hx, hy, hz]."""
    hx, hy, hz = half_extents[0], half_extents[1], half_extents[2]
    return jnp.array([hx, hy, hz, hx, hy, hz])


class DpaxCapsuleCBF:
    """CBF for capsule-capsule pairs using dpax."""

    def __init__(self, gamma: float = 5.0, margin_phi: float = 0.001):
        self.gamma = gamma
        self.margin_phi = margin_phi

        self._grad_fn = jax.jit(
            grad(proximity, argnums=(1, 2, 4, 5))
        )

        # Warm up JAX JIT
        _R = 0.1
        _a = jnp.zeros(3)
        _b = jnp.ones(3)
        _ = proximity(_R, _a, _b, _R, _a + 5.0, _b + 5.0)
        _ = self._grad_fn(_R, _a, _b, _R, _a + 5.0, _b + 5.0)

    def build_constraint(
        self,
        R1, a1, b1, J_a1, J_b1,
        R2, a2, b2, J_a2, J_b2,
        **kwargs,
    ):
        """Build capsule CBF constraint.

        Returns (phi, A_row, b_val, p1, p2).
        """
        R1_j = jnp.float64(R1)
        R2_j = jnp.float64(R2)
        a1_j = jnp.array(a1, dtype=jnp.float64)
        b1_j = jnp.array(b1, dtype=jnp.float64)
        a2_j = jnp.array(a2, dtype=jnp.float64)
        b2_j = jnp.array(b2, dtype=jnp.float64)

        phi = float(proximity(R1_j, a1_j, b1_j, R2_j, a2_j, b2_j))

        ga1, gb1, ga2, gb2 = self._grad_fn(
            R1_j, a1_j, b1_j, R2_j, a2_j, b2_j,
        )

        dphi_dq = (
            np.asarray(ga1) @ J_a1 + np.asarray(gb1) @ J_b1
            + np.asarray(ga2) @ J_a2 + np.asarray(gb2) @ J_b2
        )

        h = phi - self.margin_phi
        A_row = dphi_dq
        b_val = -self.gamma * h

        # Closest points on centerlines
        Q, q, _ = get_cost_terms(a1_j, b1_j, a2_j, b2_j)
        z = active_set_qp(Q, q)
        p1 = np.asarray(b1_j + z[0] * (a1_j - b1_j))
        p2 = np.asarray(b2_j + z[1] * (a2_j - b2_j))

        return phi, A_row, b_val, p1, p2


class SphereCBF:
    """CBF for sphere-sphere pairs with closed-form proximity.

    Uses analytical phi = ||c1-c2||^2 - (R1+R2)^2 and its gradient.
    All computation via jax.numpy for CPU/GPU transparency.
    """

    def __init__(self, gamma: float = 5.0, margin_phi: float = 0.001):
        self.gamma = gamma
        self.margin_phi = margin_phi

    def build_constraints_batch(
        self,
        R1: float, centers1: np.ndarray, jacobians1: np.ndarray,
        R2: float, centers2: np.ndarray, jacobians2: np.ndarray,
    ):
        """Build all sphere-sphere CBF constraints for one collision pair.

        Parameters
        ----------
        R1 : float
            Radius of body 1's spheres.
        centers1 : (n1, 3) array
            Sphere centers for body 1.
        jacobians1 : (n1, 3, n_q) array
            Jacobians for body 1 sphere centers.
        R2 : float
            Radius of body 2's spheres.
        centers2 : (n2, 3) array
            Sphere centers for body 2.
        jacobians2 : (n2, 3, n_q) array
            Jacobians for body 2 sphere centers.

        Returns
        -------
        constraints : list of (A_row, b_val) tuples
        closest_points : list of (p1, p2) tuples
        """
        c1 = jnp.asarray(centers1)   # (n1, 3)
        c2 = jnp.asarray(centers2)   # (n2, 3)
        J1 = jnp.asarray(jacobians1) # (n1, 3, n_q)
        J2 = jnp.asarray(jacobians2) # (n2, 3, n_q)

        # Pairwise differences: (n1, n2, 3)
        diffs = c1[:, None, :] - c2[None, :, :]

        # phi = ||c1-c2||^2 - (R1+R2)^2: (n1, n2)
        r_sum_sq = (R1 + R2) ** 2
        phis = jnp.sum(diffs ** 2, axis=2) - r_sum_sq

        # Jacobian differences: (n1, n2, 3, n_q)
        J_diffs = J1[:, None, :, :] - J2[None, :, :, :]

        # dphi/dq = 2*(c1-c2) @ (J_c1 - J_c2): (n1, n2, n_q)
        dphi_dqs = 2.0 * jnp.einsum('ijk,ijkl->ijl', diffs, J_diffs)

        # CBF constraints
        hs = phis - self.margin_phi
        b_vals = -self.gamma * hs

        # Convert to numpy and flatten
        phis_np = np.asarray(phis)
        dphi_np = np.asarray(dphi_dqs)
        b_np = np.asarray(b_vals)
        c1_np = np.asarray(c1)
        c2_np = np.asarray(c2)

        n1, n2 = phis_np.shape
        diffs_np = np.asarray(diffs)
        constraints = []
        closest_points = []
        for i in range(n1):
            for j in range(n2):
                constraints.append((dphi_np[i, j], b_np[i, j]))
                # Surface closest points: offset centers by radius along the line
                d = diffs_np[i, j]  # c1 - c2
                dist = np.linalg.norm(d)
                if dist > 1e-8:
                    direction = d / dist
                    p1 = c1_np[i] - R1 * direction
                    p2 = c2_np[j] + R2 * direction
                else:
                    p1, p2 = c1_np[i], c2_np[j]
                closest_points.append((p1, p2))

        return constraints, closest_points


class DpaxBoxCBF:
    """CBF for box-box pairs using dpax polytope proximity.

    polytope_proximity returns alpha (scaling factor).
    alpha > 1 = separated, alpha <= 1 = collision.
    """

    def __init__(self, gamma: float = 5.0, beta: float = 1.05):
        self.gamma = gamma
        self.beta = beta

        # Warm up JAX JIT with dummy boxes
        _A = _BOX_A
        _b = jnp.ones(6)
        _r1 = jnp.zeros(3)
        _Q = jnp.eye(3)
        _r2 = jnp.array([5.0, 0.0, 0.0])
        _ = polytope_proximity(_A, _b, _r1, _Q, _A, _b, _r2, _Q)
        _ = polytope_proximity_grads(_A, _b, _r1, _Q, _A, _b, _r2, _Q)

    def build_constraint(
        self,
        bodyA, centerA, rotA, J6_A,
        bodyB, centerB, rotB, J6_B,
        *, b_override_B=None,
    ):
        """Build box CBF constraint.

        Args:
            b_override_B: optional (6,) jnp array for body B halfspace b-vector.
                          When provided, bodyB is ignored.

        Returns (alpha, A_row, b_val, p1, p2).
        """
        b1 = _box_b_from_body(bodyA)
        b2 = b_override_B if b_override_B is not None else _box_b_from_body(bodyB)
        r1 = jnp.array(centerA, dtype=jnp.float64)
        Q1 = jnp.array(rotA, dtype=jnp.float64)
        r2 = jnp.array(centerB, dtype=jnp.float64)
        Q2 = jnp.array(rotB, dtype=jnp.float64)

        alpha, _, _, gr1, gQ1, _, _, gr2, gQ2 = \
            polytope_proximity_grads(
                _BOX_A, b1, r1, Q1, _BOX_A, b2, r2, Q2,
            )
        alpha = float(alpha)
        gr1 = np.asarray(gr1)
        gQ1 = np.asarray(gQ1)
        gr2 = np.asarray(gr2)
        gQ2 = np.asarray(gQ2)

        # Chain rule: dalpha/dq = dalpha/dr @ J_trans + dalpha/domega @ J_rot
        J_trans_A = J6_A[:3, :]
        J_rot_A = J6_A[3:, :]
        J_trans_B = J6_B[:3, :]
        J_rot_B = J6_B[3:, :]

        # Convert rotation matrix gradient to angular velocity gradient
        # dalpha/domega_k = sum(gQ * skew(e_k) @ Q)
        # Efficient form: S = Q @ gQ^T, dalpha/domega = vee(S - S^T)
        def _rot_grad_to_omega(gQ, Q):
            S = Q @ gQ.T
            return np.array([
                S[1, 2] - S[2, 1],
                S[2, 0] - S[0, 2],
                S[0, 1] - S[1, 0],
            ])

        domega_A = _rot_grad_to_omega(gQ1, np.asarray(Q1))
        domega_B = _rot_grad_to_omega(gQ2, np.asarray(Q2))

        dalpha_dq = (
            gr1 @ J_trans_A + domega_A @ J_rot_A
            + gr2 @ J_trans_B + domega_B @ J_rot_B
        )

        h = alpha - self.beta
        A_row = dalpha_dq
        b_val = -self.gamma * h

        # Approximate closest points as centers (no centerline concept)
        p1 = np.asarray(r1)
        p2 = np.asarray(r2)

        return alpha, A_row, b_val, p1, p2
