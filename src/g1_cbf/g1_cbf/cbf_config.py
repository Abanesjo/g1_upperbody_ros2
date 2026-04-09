"""CBFConfig subclass for G1 upper-body self-collision + human avoidance.

Uses cbfpy for the CBF-QP framework and dpax for capsule proximity.
The barrier function h_1(z) computes capsule-capsule distances for all
collision pairs. JAX autodiff handles all gradient computation.

State z = (8,) controlled joint positions.
Control u = (8,) joint velocities.
Dynamics: z_dot = u  (f=0, g=I, relative degree 1).
"""

import jax.numpy as jnp

from cbfpy import CBFConfig
from dpax.endpoints import proximity

from g1_cbf.jax_kinematics import (
    capsule_endpoints_all,
    COLLISION_PAIR_INDICES,
    N_SELF_PAIRS,
    N_BODIES,
    N_HUMAN_CAPSULES,
    RADII,
)


class G1CollisionCBFConfig(CBFConfig):
    """CBF config for G1 capsule collision avoidance.

    Barriers:
        - 9 self-collision pairs (robot body vs robot body)
        - 49 human-robot pairs (7 robot bodies × 7 human capsules), masked
          to a large safe value when fewer than 7 human capsules are active.

    Total barrier count: 58 (fixed for JAX static shapes).
    """

    def __init__(
        self,
        gamma: float = 5.0,
        margin_phi: float = 0.001,
        max_velocity: float = 2.0,
    ):
        self.gamma_val = gamma
        self.margin_phi = margin_phi

        # Dummy human capsules for cbfpy validation at init
        dummy_human = jnp.zeros((N_HUMAN_CAPSULES, 7))
        dummy_count = jnp.array(0)

        super().__init__(
            n=8,
            m=8,
            u_min=-max_velocity * jnp.ones(8),
            u_max=max_velocity * jnp.ones(8),
            relax_qp=True,
            cbf_relaxation_penalty=1e4,
            solver_tol=1e-3,
            init_args=(dummy_human, dummy_count),
        )

    def f(self, z, *args, **kwargs):
        return jnp.zeros(8)

    def g(self, z, *args, **kwargs):
        return jnp.eye(8)

    def h_1(self, z, human_capsules, human_count, **kwargs):
        """Barrier: capsule proximity for all collision pairs.

        Args:
            z: (8,) joint positions.
            human_capsules: (N_HUMAN_CAPSULES, 7) packed as [ax,ay,az, bx,by,bz, r].
            human_count: scalar, number of active human capsules.

        Returns:
            (N_SELF_PAIRS + N_BODIES * N_HUMAN_CAPSULES,) barrier values.
            Positive = safe, zero = boundary, negative = violation.
        """
        a_robot, b_robot = capsule_endpoints_all(z)

        # Self-collision barriers
        barriers = []
        for i, j in COLLISION_PAIR_INDICES:
            phi = proximity(
                RADII[i], a_robot[i], b_robot[i],
                RADII[j], a_robot[j], b_robot[j],
            )
            barriers.append(phi - self.margin_phi)

        # Human-robot barriers
        h_a = human_capsules[:, :3]
        h_b = human_capsules[:, 3:6]
        h_r = human_capsules[:, 6]

        for i in range(N_BODIES):
            for j in range(N_HUMAN_CAPSULES):
                phi = proximity(
                    RADII[i], a_robot[i], b_robot[i],
                    h_r[j], h_a[j], h_b[j],
                )
                # Mask inactive human capsules to large safe value
                masked = jnp.where(
                    j < human_count,
                    phi - self.margin_phi,
                    1e6,
                )
                barriers.append(masked)

        return jnp.array(barriers)

    def alpha(self, h, *args, **kwargs):
        return self.gamma_val * h
