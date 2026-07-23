"""CBFConfig subclass for G1 upper-body collision avoidance.

Supports two collision geometry modes:
- capsules: dpax line-segment proximity (default)
- spheres: analytical sphere-sphere distance

State z = (11,) controlled joint positions.
Control u = (11,) joint velocities.
Dynamics: z_dot = u  (f=0, g=I, relative degree 1).
"""

import jax.numpy as jnp

from cbfpy import CBFConfig
from dpax.endpoints import proximity

from g1_cbf.active_pairs import N_EXTERNAL_ROBOT_BODIES
from g1_cbf.jax_kinematics import (
    capsule_endpoints_all,
    compute_sphere_counts,
    compute_human_sphere_counts,
    sphere_centers,
    COLLISION_PAIR_INDICES,
    N_SELF_PAIRS,
    N_HUMAN_CAPSULES,
    N_LEG_JOINTS,
    N_CONTROLLED_JOINTS,
    HEAD_COLLIDER_BODY_INDEX,
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
        external_torso_margin_phi: float = None,
        external_torso_gamma: float = None,
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
        self.external_torso_margin_phi = (
            external_margin_phi
            if external_torso_margin_phi is None
            else float(external_torso_margin_phi)
        )
        self.external_torso_gamma = (
            external_gamma
            if external_torso_gamma is None
            else float(external_torso_gamma)
        )
        if self.external_torso_margin_phi < 0.0:
            raise ValueError('external_torso_margin_phi must be non-negative')
        if self.external_torso_gamma <= 0.0:
            raise ValueError('external_torso_gamma must be positive')
        self.geom = str(collision_geometry).lower()
        if self.geom not in ('capsules', 'spheres'):
            raise ValueError(
                "collision_geometry must be 'capsules' or 'spheres'"
            )
        self.radius_gain = sphere_radius_gain
        self.external_pair_slots = int(
            external_pair_slots or (
                N_EXTERNAL_ROBOT_BODIES * N_HUMAN_CAPSULES
            )
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
        self.num_head_circle_barriers = 1

        self.alpha_gains = jnp.concatenate([
            internal_gamma * jnp.ones(self.num_internal_barriers),
            external_gamma * jnp.ones(self.num_external_barriers),
            external_gamma * jnp.ones(self.num_head_circle_barriers),
        ])

        # Dummy args for cbfpy validation
        dummy_legs = jnp.zeros(N_LEG_JOINTS)
        dummy_human = jnp.zeros((N_HUMAN_CAPSULES, 7))
        dummy_pair_indices = jnp.zeros((self.external_pair_slots, 2), dtype=jnp.int32)
        dummy_pair_mask = jnp.zeros(self.external_pair_slots, dtype=bool)
        dummy_internal_indices = jnp.zeros((self.internal_pair_slots, 4), dtype=jnp.int32)
        dummy_internal_mask = jnp.zeros(self.internal_pair_slots, dtype=bool)
        dummy_pelvis_position = jnp.zeros(3)
        dummy_pelvis_quat = jnp.array([0.0, 0.0, 0.0, 1.0])
        dummy_workspace_center_xy = jnp.zeros(2)
        dummy_world_circle_radius = jnp.array(3.0)
        dummy_head_collider_radius = jnp.array(0.3)
        dummy_head_circle_enabled = jnp.array(False)

        super().__init__(
            n=N_CONTROLLED_JOINTS,
            m=N_CONTROLLED_JOINTS,
            u_min=-max_velocity * jnp.ones(N_CONTROLLED_JOINTS),
            u_max=max_velocity * jnp.ones(N_CONTROLLED_JOINTS),
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
                dummy_pelvis_position,
                dummy_pelvis_quat,
                dummy_workspace_center_xy,
                dummy_world_circle_radius,
                dummy_head_collider_radius,
                dummy_head_circle_enabled,
            ),
        )

    def f(self, z, *args, **kwargs):
        return jnp.zeros(N_CONTROLLED_JOINTS)

    def g(self, z, *args, **kwargs):
        return jnp.eye(N_CONTROLLED_JOINTS)

    def h_1(self, z, q_legs, human_capsules, active_pair_indices,
            active_pair_mask, active_internal_indices, active_internal_mask,
            pelvis_position, pelvis_quat, workspace_center_xy,
            world_circle_radius, head_collider_radius, head_circle_enabled,
            **kwargs):
        if self.geom == 'spheres':
            return self._h1_spheres(
                z, q_legs, human_capsules, active_pair_indices,
                active_pair_mask, active_internal_indices, active_internal_mask,
                pelvis_position, pelvis_quat, workspace_center_xy,
                world_circle_radius, head_collider_radius,
                head_circle_enabled,
            )
        else:
            return self._h1_capsules(
                z, q_legs, human_capsules, active_pair_indices,
                active_pair_mask, pelvis_position, pelvis_quat,
                workspace_center_xy, world_circle_radius,
                head_collider_radius, head_circle_enabled,
            )

    def alpha(self, h, q_legs=None, human_capsules=None,
              active_pair_indices=None, active_pair_mask=None,
              active_internal_indices=None, active_internal_mask=None,
              pelvis_position=None, pelvis_quat=None,
              workspace_center_xy=None,
              world_circle_radius=None, head_collider_radius=None,
              head_circle_enabled=None, **kwargs):
        del (
            q_legs, human_capsules, active_pair_mask,
            active_internal_indices, active_internal_mask, pelvis_position,
            pelvis_quat, workspace_center_xy, world_circle_radius,
            head_collider_radius, head_circle_enabled, kwargs,
        )
        if active_pair_indices is None:
            return self.alpha_gains * h
        return self._alpha_gains_for_pairs(active_pair_indices) * h

    def _alpha_gains_for_pairs(self, active_pair_indices):
        external_pair_gains = jnp.where(
            active_pair_indices[:, 0] == HEAD_COLLIDER_BODY_INDEX,
            self.external_torso_gamma,
            self.external_gamma,
        )
        if self.geom == 'spheres':
            external_gains = jnp.repeat(
                external_pair_gains,
                self.max_robot_spheres * self.max_human_spheres,
            )
        else:
            external_gains = external_pair_gains
        return jnp.concatenate([
            self.internal_gamma * jnp.ones(self.num_internal_barriers),
            external_gains,
            self.external_gamma * jnp.ones(self.num_head_circle_barriers),
        ])

    # ------------------------------------------------------------------
    # Capsule mode
    # ------------------------------------------------------------------

    def _h1_capsules(self, z, q_legs, human_capsules, active_pair_indices,
                     active_pair_mask, pelvis_position, pelvis_quat,
                     workspace_center_xy, world_circle_radius,
                     head_collider_radius, head_circle_enabled):
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
            margin = jnp.where(
                i == HEAD_COLLIDER_BODY_INDEX,
                self.external_torso_margin_phi,
                self.external_margin_phi,
            )
            phi = self._capsule_barrier(
                RADII[i], a_robot[i], b_robot[i],
                h_r[j], h_a[j], h_b[j],
                margin,
            )
            barriers.append(jnp.where(
                active_pair_mask[slot], phi, 1.0,
            ))

        barriers.append(self._head_circle_barrier(
            a_robot[HEAD_COLLIDER_BODY_INDEX],
            pelvis_position,
            pelvis_quat,
            workspace_center_xy,
            world_circle_radius,
            head_collider_radius,
            head_circle_enabled,
        ))

        return jnp.array(barriers)

    # ------------------------------------------------------------------
    # Sphere mode
    # ------------------------------------------------------------------

    def _h1_spheres(self, z, q_legs, human_capsules, active_pair_indices,
                    active_pair_mask, active_internal_indices,
                    active_internal_mask, pelvis_position, pelvis_quat,
                    workspace_center_xy, world_circle_radius,
                    head_collider_radius, head_circle_enabled):
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
            margin = jnp.where(
                i == HEAD_COLLIDER_BODY_INDEX,
                self.external_torso_margin_phi,
                self.external_margin_phi,
            )
            r_sum = ri + h_r[j] * rg + margin
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

        barriers.append(self._head_circle_barrier(
            a_robot[HEAD_COLLIDER_BODY_INDEX],
            pelvis_position,
            pelvis_quat,
            workspace_center_xy,
            world_circle_radius,
            head_collider_radius,
            head_circle_enabled,
        ))

        return jnp.array(barriers)

    def _head_circle_barrier(self, head_center_pelvis, pelvis_position,
                             pelvis_quat, workspace_center_xy,
                             world_circle_radius, head_collider_radius,
                             head_circle_enabled):
        head_center_world = (
            pelvis_position
            + self._quat_rotate(pelvis_quat, head_center_pelvis)
        )
        safe_radius = (
            world_circle_radius
            - head_collider_radius
            - self.external_margin_phi
        )
        horizontal_dist_sq = jnp.sum(
            (head_center_world[:2] - workspace_center_xy) ** 2
        )
        phi = safe_radius ** 2 - horizontal_dist_sq
        return jnp.where(head_circle_enabled, phi, 1.0)

    @staticmethod
    def _quat_rotate(quat_xyzw, vec):
        quat_xyzw = quat_xyzw / jnp.maximum(
            jnp.linalg.norm(quat_xyzw),
            1e-9,
        )
        q_vec = quat_xyzw[:3]
        q_w = quat_xyzw[3]
        t = 2.0 * jnp.cross(q_vec, vec)
        return vec + q_w * t + jnp.cross(q_vec, t)

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


class G1CmdVelCBFConfig(CBFConfig):
    """CBF config for planar base velocity safety constraints."""

    def __init__(
        self,
        external_gamma: float = 5.0,
        external_margin_phi: float = 0.001,
        lin_vel_x_limits: list = None,
        lin_vel_y_limits: list = None,
        solver_tol: float = 1e-3,
    ):
        if external_gamma <= 0.0:
            raise ValueError('external_gamma must be positive')
        if external_margin_phi < 0.0:
            raise ValueError('external_margin_phi must be non-negative')

        lin_vel_x_limits = lin_vel_x_limits or [-0.5, 1.0]
        lin_vel_y_limits = lin_vel_y_limits or [-0.5, 0.5]
        if len(lin_vel_x_limits) != 2 or len(lin_vel_y_limits) != 2:
            raise ValueError('linear velocity limits must have [min, max]')
        if lin_vel_x_limits[0] > lin_vel_x_limits[1]:
            raise ValueError('lin_vel_x_limits min must be <= max')
        if lin_vel_y_limits[0] > lin_vel_y_limits[1]:
            raise ValueError('lin_vel_y_limits min must be <= max')

        self.external_gamma = external_gamma
        self.external_margin_phi = external_margin_phi
        self.num_human_endpoint_barriers = 2 * N_HUMAN_CAPSULES
        self.alpha_gains = external_gamma * jnp.ones(
            1 + self.num_human_endpoint_barriers,
        )

        dummy_pelvis_quat = jnp.array([0.0, 0.0, 0.0, 1.0])
        dummy_workspace_center_xy = jnp.zeros(2)
        dummy_world_circle_radius = jnp.array(3.0)
        dummy_head_collider_radius = jnp.array(0.3)
        dummy_cbf_enabled = jnp.array(True)
        dummy_human_endpoint_points_xy = jnp.zeros((
            self.num_human_endpoint_barriers,
            2,
        ))
        dummy_human_endpoint_radii = jnp.zeros(
            self.num_human_endpoint_barriers,
        )
        dummy_human_endpoint_mask = jnp.zeros(
            self.num_human_endpoint_barriers,
            dtype=bool,
        )

        super().__init__(
            n=2,
            m=2,
            u_min=jnp.array([lin_vel_x_limits[0], lin_vel_y_limits[0]]),
            u_max=jnp.array([lin_vel_x_limits[1], lin_vel_y_limits[1]]),
            relax_qp=True,
            cbf_relaxation_penalty=1e4,
            solver_tol=solver_tol,
            init_args=(
                dummy_pelvis_quat,
                dummy_workspace_center_xy,
                dummy_world_circle_radius,
                dummy_head_collider_radius,
                dummy_cbf_enabled,
                dummy_human_endpoint_points_xy,
                dummy_human_endpoint_radii,
                dummy_human_endpoint_mask,
            ),
        )

    def f(self, z, *args, **kwargs):
        return jnp.zeros(2)

    def g(self, z, pelvis_quat, *args, **kwargs):
        body_x_world = self._quat_rotate(
            pelvis_quat,
            jnp.array([1.0, 0.0, 0.0]),
        )[:2]
        body_y_world = self._quat_rotate(
            pelvis_quat,
            jnp.array([0.0, 1.0, 0.0]),
        )[:2]
        return jnp.stack([body_x_world, body_y_world], axis=1)

    def h_1(self, z, pelvis_quat, workspace_center_xy,
            world_circle_radius, head_collider_radius, cbf_enabled,
            human_endpoint_points_xy, human_endpoint_radii,
            human_endpoint_mask, **kwargs):
        del pelvis_quat
        world_safe_radius = (
            world_circle_radius
            - head_collider_radius
            - self.external_margin_phi
        )
        world_phi = (
            world_safe_radius ** 2
            - jnp.sum((z - workspace_center_xy) ** 2)
        )

        endpoint_safe_radii = (
            head_collider_radius
            + human_endpoint_radii
            + self.external_margin_phi
        )
        endpoint_delta = z - human_endpoint_points_xy
        endpoint_phi = (
            jnp.sum(endpoint_delta ** 2, axis=1)
            - endpoint_safe_radii ** 2
        )
        endpoint_phi = jnp.where(
            cbf_enabled & human_endpoint_mask,
            endpoint_phi,
            1.0,
        )

        return jnp.concatenate([
            jnp.array([jnp.where(cbf_enabled, world_phi, 1.0)]),
            endpoint_phi,
        ])

    def alpha(self, h, *args, **kwargs):
        return self.alpha_gains * h

    @staticmethod
    def _quat_rotate(quat_xyzw, vec):
        quat_xyzw = quat_xyzw / jnp.maximum(
            jnp.linalg.norm(quat_xyzw),
            1e-9,
        )
        q_vec = quat_xyzw[:3]
        q_w = quat_xyzw[3]
        t = 2.0 * jnp.cross(q_vec, vec)
        return vec + q_w * t + jnp.cross(q_vec, t)
