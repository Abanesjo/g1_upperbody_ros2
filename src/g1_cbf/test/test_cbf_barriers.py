import jax
import jax.numpy as jnp
import numpy as np
import pytest

from g1_cbf.active_pairs import EXTERNAL_ROBOT_BODY_INDICES
from g1_cbf.cbf_config import G1CollisionCBFConfig
from g1_cbf.jax_kinematics import (
    COLLISION_PAIR_INDICES,
    CONTROLLED_JOINT_DEFAULTS,
    HEAD_COLLIDER_BODY_INDEX,
    N_HUMAN_CAPSULES,
    N_LEG_JOINTS,
    RADII,
    capsule_endpoints_all,
    compute_human_sphere_counts,
)


PAIR_SLOTS = 12
# One human's capsules, as in config/params.yaml.
HUMAN_HALF_LENGTHS = [
    0.33, 0.20, 0.20, 0.145, 0.145, 0.225, 0.225, 0.16, 0.16, 0.225, 0.225,
]
HUMAN_RADII = [
    0.10, 0.05, 0.05, 0.05, 0.05, 0.065, 0.065, 0.10, 0.10, 0.065, 0.065,
]


def _scenario(seed):
    rng = np.random.default_rng(seed)
    z = jnp.array(
        CONTROLLED_JOINT_DEFAULTS
        + rng.uniform(-0.3, 0.3, CONTROLLED_JOINT_DEFAULTS.shape)
    )
    q_legs = jnp.array(rng.uniform(-0.2, 0.2, N_LEG_JOINTS))
    centers = rng.uniform(
        [-0.2, -0.8, -0.4], [1.0, 0.8, 0.6], (N_HUMAN_CAPSULES, 3),
    )
    half_axes = rng.uniform(-0.15, 0.15, (N_HUMAN_CAPSULES, 3))
    human_capsules = jnp.array(np.column_stack([
        centers + half_axes,
        centers - half_axes,
        rng.uniform(0.05, 0.1, N_HUMAN_CAPSULES),
    ]))
    robot_bodies = rng.choice(
        np.array(EXTERNAL_ROBOT_BODY_INDICES), PAIR_SLOTS,
    )
    # The torso pair uses external_torso_margin_phi instead.
    robot_bodies[0] = HEAD_COLLIDER_BODY_INDEX
    pair_indices = np.column_stack([
        robot_bodies, rng.integers(0, N_HUMAN_CAPSULES, PAIR_SLOTS),
    ]).astype(np.int32)
    pair_mask = rng.random(PAIR_SLOTS) < 0.7
    pair_mask[:2] = [True, False]
    return z, q_legs, human_capsules, pair_indices, pair_mask


def _per_pair_barriers(config, z, q_legs, human_capsules, pair_indices,
                       pair_mask):
    """Self-collision and external barriers, one pair at a time."""
    a, b = capsule_endpoints_all(z, q_legs)
    barriers = [
        config._capsule_barrier(
            RADII[i], a[i], b[i], RADII[j], a[j], b[j],
            config.internal_margin_phi,
        )
        for i, j in COLLISION_PAIR_INDICES
    ]
    for (i, j), active in zip(pair_indices.tolist(), pair_mask.tolist()):
        margin = (
            config.external_torso_margin_phi
            if i == HEAD_COLLIDER_BODY_INDEX
            else config.external_margin_phi
        )
        phi = config._capsule_barrier(
            RADII[i], a[i], b[i],
            human_capsules[j, 6], human_capsules[j, :3],
            human_capsules[j, 3:6],
            margin,
        )
        barriers.append(phi if active else jnp.array(1.0))
    return jnp.stack(barriers)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_vectorized_capsule_barriers_match_the_per_pair_loop(seed):
    config = G1CollisionCBFConfig(
        internal_margin_phi=0.08,
        external_margin_phi=0.5,
        external_torso_margin_phi=0.1,
        external_pair_slots=PAIR_SLOTS,
    )
    z, q_legs, human_capsules, pair_indices, pair_mask = _scenario(seed)
    head_args = (
        jnp.array([0.1, -0.2, 0.8]),
        jnp.array([0.0, 0.0, np.sin(0.2), np.cos(0.2)]),
        jnp.array([0.3, 0.1]),
        jnp.array(1.5),
        jnp.array(0.1),
        jnp.array(True),
    )

    def barriers(zz):
        return config.h_1(
            zz, q_legs, human_capsules,
            jnp.array(pair_indices), jnp.array(pair_mask),
            jnp.zeros((1, 4), dtype=jnp.int32), jnp.zeros(1, dtype=bool),
            *head_args,
        )

    def per_pair(zz):
        return _per_pair_barriers(
            config, zz, q_legs, human_capsules, pair_indices, pair_mask,
        )

    h = barriers(z)
    assert h.shape == (len(COLLISION_PAIR_INDICES) + PAIR_SLOTS + 1,)
    np.testing.assert_allclose(h[:-1], per_pair(z), rtol=1e-9, atol=1e-12)
    a, _ = capsule_endpoints_all(z, q_legs)
    head = config._head_circle_barrier(
        a[HEAD_COLLIDER_BODY_INDEX], *head_args,
    )
    np.testing.assert_allclose(h[-1], head, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        jax.jacfwd(barriers)(z)[:-1], jax.jacfwd(per_pair)(z),
        rtol=1e-9, atol=1e-12,
    )


def test_sphere_counts_repeat_one_human_for_every_human():
    config = G1CollisionCBFConfig(
        collision_geometry='spheres',
        human_half_lengths=HUMAN_HALF_LENGTHS,
        human_radii=HUMAN_RADII,
        external_pair_slots=2,
        internal_pair_slots=2,
    )
    one_human = compute_human_sphere_counts(HUMAN_HALF_LENGTHS, HUMAN_RADII)

    assert config.human_sphere_counts == [
        one_human[k % len(one_human)] for k in range(N_HUMAN_CAPSULES)
    ]
    assert config.max_human_spheres == max(one_human)
