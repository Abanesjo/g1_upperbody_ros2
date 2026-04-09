"""CBFConfig subclass for G1 upper-body collision avoidance.

Supports three collision geometry modes:
- capsules: dpax line-segment proximity (default)
- spheres: analytical sphere-sphere distance
- boxes: dpax polytope proximity

State z = (8,) controlled joint positions.
Control u = (8,) joint velocities.
Dynamics: z_dot = u  (f=0, g=I, relative degree 1).
"""

import jax.numpy as jnp

from cbfpy import CBFConfig
from dpax.endpoints import proximity
from dpax.polytopes import polytope_proximity

from g1_cbf.jax_kinematics import (
    capsule_endpoints_all,
    fk_body_transforms,
    compute_sphere_counts,
    compute_max_human_spheres,
    sphere_centers,
    capsule_to_box_params,
    COLLISION_PAIR_INDICES,
    N_SELF_PAIRS,
    N_BODIES,
    N_HUMAN_CAPSULES,
    N_LEG_JOINTS,
    HALF_LENGTHS,
    RADII,
    BOX_A,
    BOX_B,
)


class G1CollisionCBFConfig(CBFConfig):
    """CBF config for G1 collision avoidance with capsule/sphere/box modes."""

    def __init__(
        self,
        gamma: float = 5.0,
        margin_phi: float = 0.001,
        max_velocity: float = 2.0,
        collision_geometry: str = 'capsules',
        sphere_interpolation_level: int = 0,
        sphere_radius_gain: float = 1.0,
        beta: float = 1.05,
        solver_tol: float = 1e-3,
    ):
        self.gamma_val = gamma
        self.margin_phi = margin_phi
        self.geom = collision_geometry
        self.radius_gain = sphere_radius_gain
        self.beta_val = beta

        # Compute barrier count based on geometry mode
        if self.geom == 'spheres':
            self.sphere_counts = compute_sphere_counts(sphere_interpolation_level)
            self.human_spheres_per_cap = compute_max_human_spheres(sphere_interpolation_level)
            n_self = sum(
                self.sphere_counts[i] * self.sphere_counts[j]
                for i, j in COLLISION_PAIR_INDICES
            )
            # Human: each robot body's spheres × each human capsule's spheres
            n_human = sum(self.sphere_counts[i] for i in range(N_BODIES)) \
                * self.human_spheres_per_cap * N_HUMAN_CAPSULES
        else:
            # Capsules and boxes: 1 barrier per pair
            n_self = N_SELF_PAIRS
            n_human = N_BODIES * N_HUMAN_CAPSULES

        # Dummy args for cbfpy validation
        dummy_legs = jnp.zeros(N_LEG_JOINTS)
        dummy_human = jnp.zeros((N_HUMAN_CAPSULES, 7))
        dummy_count = jnp.array(0)

        super().__init__(
            n=8,
            m=8,
            u_min=-max_velocity * jnp.ones(8),
            u_max=max_velocity * jnp.ones(8),
            relax_qp=True,
            cbf_relaxation_penalty=1e4,
            solver_tol=solver_tol,
            init_args=(dummy_legs, dummy_human, dummy_count),
        )

    def f(self, z, *args, **kwargs):
        return jnp.zeros(8)

    def g(self, z, *args, **kwargs):
        return jnp.eye(8)

    def h_1(self, z, q_legs, human_capsules, human_count, **kwargs):
        if self.geom == 'spheres':
            return self._h1_spheres(z, q_legs, human_capsules, human_count)
        elif self.geom == 'boxes':
            return self._h1_boxes(z, q_legs, human_capsules, human_count)
        else:
            return self._h1_capsules(z, q_legs, human_capsules, human_count)

    def alpha(self, h, *args, **kwargs):
        return self.gamma_val * h

    # ------------------------------------------------------------------
    # Capsule mode
    # ------------------------------------------------------------------

    def _h1_capsules(self, z, q_legs, human_capsules, human_count):
        a_robot, b_robot = capsule_endpoints_all(z, q_legs)
        barriers = []

        for i, j in COLLISION_PAIR_INDICES:
            phi = proximity(
                RADII[i], a_robot[i], b_robot[i],
                RADII[j], a_robot[j], b_robot[j],
            )
            barriers.append(phi - self.margin_phi)

        h_a = human_capsules[:, :3]
        h_b = human_capsules[:, 3:6]
        h_r = human_capsules[:, 6]

        for i in range(N_BODIES):
            for j in range(N_HUMAN_CAPSULES):
                phi = proximity(
                    RADII[i], a_robot[i], b_robot[i],
                    h_r[j], h_a[j], h_b[j],
                )
                barriers.append(jnp.where(
                    j < human_count, phi - self.margin_phi, 1.0,
                ))

        return jnp.array(barriers)

    # ------------------------------------------------------------------
    # Sphere mode
    # ------------------------------------------------------------------

    def _h1_spheres(self, z, q_legs, human_capsules, human_count):
        a_robot, b_robot = capsule_endpoints_all(z, q_legs)
        rg = self.radius_gain
        barriers = []

        # Self-collision: sphere-sphere pairs
        for i, j in COLLISION_PAIR_INDICES:
            ci = sphere_centers(a_robot[i], b_robot[i], self.sphere_counts[i])
            cj = sphere_centers(a_robot[j], b_robot[j], self.sphere_counts[j])
            ri = RADII[i] * rg
            rj = RADII[j] * rg
            r_sum_sq = (ri + rj) ** 2
            for si in range(self.sphere_counts[i]):
                for sj in range(self.sphere_counts[j]):
                    d_sq = jnp.sum((ci[si] - cj[sj]) ** 2)
                    barriers.append(d_sq - r_sum_sq - self.margin_phi)

        # Human-robot: decompose both into spheres, check all pairs
        h_a = human_capsules[:, :3]
        h_b = human_capsules[:, 3:6]
        h_r = human_capsules[:, 6]
        n_hs = self.human_spheres_per_cap

        for i in range(N_BODIES):
            ci = sphere_centers(a_robot[i], b_robot[i], self.sphere_counts[i])
            ri = RADII[i] * rg
            for j in range(N_HUMAN_CAPSULES):
                # Decompose human capsule j into spheres
                hj_centers = sphere_centers(h_a[j], h_b[j], n_hs)
                r_sum_sq = (ri + h_r[j] * rg) ** 2
                for si in range(self.sphere_counts[i]):
                    for sj in range(n_hs):
                        d_sq = jnp.sum((ci[si] - hj_centers[sj]) ** 2)
                        barriers.append(jnp.where(
                            j < human_count,
                            d_sq - r_sum_sq - self.margin_phi,
                            1.0,
                        ))

        return jnp.array(barriers)

    # ------------------------------------------------------------------
    # Box mode
    # ------------------------------------------------------------------

    def _h1_boxes(self, z, q_legs, human_capsules, human_count):
        transforms = fk_body_transforms(z, q_legs)
        barriers = []

        for i, j in COLLISION_PAIR_INDICES:
            Ti, Tj = transforms[i], transforms[j]
            alpha = polytope_proximity(
                BOX_A, BOX_B[i], Ti[:3, 3], Ti[:3, :3],
                BOX_A, BOX_B[j], Tj[:3, 3], Tj[:3, :3],
            )
            barriers.append(alpha - self.beta_val)

        # Human boxes
        h_a = human_capsules[:, :3]
        h_b = human_capsules[:, 3:6]
        h_r = human_capsules[:, 6]

        for i in range(N_BODIES):
            Ti = transforms[i]
            for j in range(N_HUMAN_CAPSULES):
                h_center, h_rot, h_bvec = capsule_to_box_params(
                    h_a[j], h_b[j], h_r[j],
                )
                alpha = polytope_proximity(
                    BOX_A, BOX_B[i], Ti[:3, 3], Ti[:3, :3],
                    BOX_A, h_bvec, h_center, h_rot,
                )
                barriers.append(jnp.where(
                    j < human_count, alpha - self.beta_val, 1.0,
                ))

        return jnp.array(barriers)
