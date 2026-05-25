"""CBFConfig subclass for G1 upper-body collision avoidance.

Supports two collision geometry modes:
- capsules: dpax line-segment proximity (default)
- spheres: analytical sphere-sphere distance

State z = (8,) controlled joint positions.
Control u = (8,) joint velocities.
Dynamics: z_dot = u  (f=0, g=I, relative degree 1).
"""

import jax.numpy as jnp

from cbfpy import CBFConfig
from dpax.endpoints import proximity

from g1_cbf.jax_kinematics import (
    capsule_endpoints_all,
    compute_sphere_counts,
    compute_human_sphere_counts,
    sphere_centers,
    COLLISION_PAIR_INDICES,
    N_SELF_PAIRS,
    N_BODIES,
    N_HUMAN_CAPSULES,
    N_LEG_JOINTS,
    RADII,
)


class G1CollisionCBFConfig(CBFConfig):
    """CBF config for G1 collision avoidance with capsule/sphere modes."""

    def __init__(
        self,
        internal_gamma: float = 5.0,
        external_gamma: float = 5.0,
        internal_margin_phi: float = 0.001,
        external_margin_phi: float = 0.001,
        max_velocity: float = 2.0,
        collision_geometry: str = 'capsules',
        sphere_interpolation_level: int = 0,
        sphere_radius_gain: float = 1.0,
        solver_tol: float = 1e-3,
        human_half_lengths: list = None,
        human_radii: list = None,
        external_pair_slots: int = None,
        internal_pair_slots: int = None,
    ):
        if internal_gamma <= 0.0 or external_gamma <= 0.0:
            raise ValueError('internal_gamma and external_gamma must be positive')
        if internal_margin_phi < 0.0 or external_margin_phi < 0.0:
            raise ValueError(
                'internal_margin_phi and external_margin_phi must be non-negative'
            )

        self.internal_gamma = internal_gamma
        self.external_gamma = external_gamma
        self.internal_margin_phi = internal_margin_phi
        self.external_margin_phi = external_margin_phi
        self.geom = str(collision_geometry).lower()
        if self.geom not in ('capsules', 'spheres'):
            raise ValueError(
                "collision_geometry must be 'capsules' or 'spheres'"
            )
        self.radius_gain = sphere_radius_gain
        self.external_pair_slots = int(
            external_pair_slots or (N_BODIES * N_HUMAN_CAPSULES)
        )
        self.internal_pair_slots = 1

        # Compute barrier count based on geometry mode
        if self.geom == 'spheres':
            self.sphere_counts = compute_sphere_counts(sphere_interpolation_level)
            human_counts = compute_human_sphere_counts(
                human_half_lengths or [], human_radii or [],
                sphere_interpolation_level,
            )
            if len(human_counts) < N_HUMAN_CAPSULES:
                human_counts.extend([1] * (N_HUMAN_CAPSULES - len(human_counts)))
            self.human_sphere_counts = human_counts[:N_HUMAN_CAPSULES]
            self.sphere_counts_jnp = jnp.array(self.sphere_counts)
            self.human_sphere_counts_jnp = jnp.array(self.human_sphere_counts)
            self.max_robot_spheres = max(self.sphere_counts)
            self.max_human_spheres = max(self.human_sphere_counts)
            full_internal_pairs = sum(
                self.sphere_counts[i] * self.sphere_counts[j]
                for i, j in COLLISION_PAIR_INDICES
            )
            self.internal_pair_slots = int(
                internal_pair_slots or full_internal_pairs
            )
            self.num_internal_barriers = self.internal_pair_slots
            self.num_external_barriers = (
                self.external_pair_slots
                * self.max_robot_spheres
                * self.max_human_spheres
            )
        else:
            self.num_internal_barriers = N_SELF_PAIRS
            self.num_external_barriers = self.external_pair_slots

        self.alpha_gains = jnp.concatenate([
            internal_gamma * jnp.ones(self.num_internal_barriers),
            external_gamma * jnp.ones(self.num_external_barriers),
        ])

        # Dummy args for cbfpy validation
        dummy_legs = jnp.zeros(N_LEG_JOINTS)
        dummy_human = jnp.zeros((N_HUMAN_CAPSULES, 7))
        dummy_pair_indices = jnp.zeros((self.external_pair_slots, 2), dtype=jnp.int32)
        dummy_pair_mask = jnp.zeros(self.external_pair_slots, dtype=bool)
        dummy_internal_indices = jnp.zeros((self.internal_pair_slots, 4), dtype=jnp.int32)
        dummy_internal_mask = jnp.zeros(self.internal_pair_slots, dtype=bool)

        super().__init__(
            n=8,
            m=8,
            u_min=-max_velocity * jnp.ones(8),
            u_max=max_velocity * jnp.ones(8),
            relax_qp=True,
            cbf_relaxation_penalty=1e4,
            solver_tol=solver_tol,
            init_args=(
                dummy_legs,
                dummy_human,
                dummy_pair_indices,
                dummy_pair_mask,
                dummy_internal_indices,
                dummy_internal_mask,
            ),
        )

    def f(self, z, *args, **kwargs):
        return jnp.zeros(8)

    def g(self, z, *args, **kwargs):
        return jnp.eye(8)

    def h_1(self, z, q_legs, human_capsules, active_pair_indices,
            active_pair_mask, active_internal_indices, active_internal_mask,
            **kwargs):
        if self.geom == 'spheres':
            return self._h1_spheres(
                z, q_legs, human_capsules, active_pair_indices,
                active_pair_mask, active_internal_indices, active_internal_mask,
            )
        else:
            return self._h1_capsules(
                z, q_legs, human_capsules, active_pair_indices,
                active_pair_mask,
            )

    def alpha(self, h, *args, **kwargs):
        return self.alpha_gains * h

    # ------------------------------------------------------------------
    # Capsule mode
    # ------------------------------------------------------------------

    def _h1_capsules(self, z, q_legs, human_capsules, active_pair_indices,
                     active_pair_mask):
        a_robot, b_robot = capsule_endpoints_all(z, q_legs)
        barriers = []

        for i, j in COLLISION_PAIR_INDICES:
            phi = self._capsule_barrier(
                RADII[i], a_robot[i], b_robot[i],
                RADII[j], a_robot[j], b_robot[j],
                self.internal_margin_phi,
            )
            barriers.append(phi)

        h_a = human_capsules[:, :3]
        h_b = human_capsules[:, 3:6]
        h_r = human_capsules[:, 6]

        for slot in range(self.external_pair_slots):
            i = active_pair_indices[slot, 0]
            j = active_pair_indices[slot, 1]
            phi = self._capsule_barrier(
                RADII[i], a_robot[i], b_robot[i],
                h_r[j], h_a[j], h_b[j],
                self.external_margin_phi,
            )
            barriers.append(jnp.where(
                active_pair_mask[slot], phi, 1.0,
            ))

        return jnp.array(barriers)

    # ------------------------------------------------------------------
    # Sphere mode
    # ------------------------------------------------------------------

    def _h1_spheres(self, z, q_legs, human_capsules, active_pair_indices,
                    active_pair_mask, active_internal_indices,
                    active_internal_mask):
        a_robot, b_robot = capsule_endpoints_all(z, q_legs)
        rg = self.radius_gain
        barriers = []

        # Human-robot: decompose both into spheres, check all pairs
        h_a = human_capsules[:, :3]
        h_b = human_capsules[:, 3:6]
        h_r = human_capsules[:, 6]
        robot_centers = self._padded_sphere_centers(
            a_robot, b_robot, self.sphere_counts, self.max_robot_spheres,
        )
        human_centers = self._padded_sphere_centers(
            h_a, h_b, self.human_sphere_counts, self.max_human_spheres,
        )

        # Self-collision: selected internal sphere-sphere pairs
        for slot in range(self.internal_pair_slots):
            i = active_internal_indices[slot, 0]
            si = active_internal_indices[slot, 1]
            j = active_internal_indices[slot, 2]
            sj = active_internal_indices[slot, 3]
            d_sq = jnp.sum((robot_centers[i, si] - robot_centers[j, sj]) ** 2)
            r_sum = (RADII[i] + RADII[j]) * rg + self.internal_margin_phi
            barriers.append(jnp.where(
                active_internal_mask[slot],
                d_sq - r_sum ** 2,
                1.0,
            ))

        for slot in range(self.external_pair_slots):
            i = active_pair_indices[slot, 0]
            j = active_pair_indices[slot, 1]
            ci = robot_centers[i]
            hj_centers = human_centers[j]
            ri = RADII[i] * rg
            r_sum = ri + h_r[j] * rg + self.external_margin_phi
            n_robot = self.sphere_counts_jnp[i]
            n_human = self.human_sphere_counts_jnp[j]
            for si in range(self.max_robot_spheres):
                for sj in range(self.max_human_spheres):
                    d_sq = jnp.sum((ci[si] - hj_centers[sj]) ** 2)
                    valid = (
                        active_pair_mask[slot]
                        & (si < n_robot)
                        & (sj < n_human)
                    )
                    barriers.append(jnp.where(
                        valid,
                        d_sq - r_sum ** 2,
                        1.0,
                    ))

        return jnp.array(barriers)

    @staticmethod
    def _padded_sphere_centers(a_all, b_all, counts, max_count):
        centers = []
        for i, count in enumerate(counts):
            c = sphere_centers(a_all[i], b_all[i], count)
            if count < max_count:
                pad = jnp.zeros((max_count - count, 3), dtype=c.dtype)
                c = jnp.concatenate([c, pad], axis=0)
            centers.append(c)
        return jnp.stack(centers)

    @staticmethod
    def _capsule_barrier(r1, a1, b1, r2, a2, b2, margin):
        expanded = 0.5 * margin
        return proximity(r1 + expanded, a1, b1, r2 + expanded, a2, b2)
