"""JAX broadphase selection for external robot-human collision pairs."""

import jax.numpy as jnp

from g1_cbf.jax_kinematics import (
    N_BODIES,
    N_HUMAN_CAPSULES,
    RADII,
    capsule_endpoints_all,
)


ALL_EXTERNAL_PAIR_INDICES = jnp.array(
    [
        (robot_idx, human_idx)
        for human_idx in range(N_HUMAN_CAPSULES)
        for robot_idx in range(N_BODIES)
    ],
    dtype=jnp.int32,
)


def _segment_distances(a1, b1, a2, b2):
    """Vectorized closest distances between segment arrays."""
    d1 = b1 - a1
    d2 = b2 - a2
    r = a1 - a2

    a = jnp.sum(d1 * d1, axis=1)
    e = jnp.sum(d2 * d2, axis=1)
    f = jnp.sum(d2 * r, axis=1)
    c = jnp.sum(d1 * r, axis=1)
    b_val = jnp.sum(d1 * d2, axis=1)

    eps = 1e-8
    denom = a * e - b_val * b_val

    s_general = jnp.where(
        jnp.abs(denom) > eps,
        jnp.clip((b_val * f - c * e) / denom, 0.0, 1.0),
        0.0,
    )
    t_general = (b_val * s_general + f) / jnp.maximum(e, eps)

    s_tlow = jnp.clip(-c / jnp.maximum(a, eps), 0.0, 1.0)
    s_thigh = jnp.clip((b_val - c) / jnp.maximum(a, eps), 0.0, 1.0)
    s_general = jnp.where(t_general < 0.0, s_tlow, s_general)
    s_general = jnp.where(t_general > 1.0, s_thigh, s_general)
    t_general = jnp.clip(t_general, 0.0, 1.0)

    s_degenerate_d2 = jnp.clip(-c / jnp.maximum(a, eps), 0.0, 1.0)
    s = jnp.where(e < eps, s_degenerate_d2, s_general)
    t = jnp.where(e < eps, 0.0, t_general)

    t_degenerate_d1 = jnp.clip(f / jnp.maximum(e, eps), 0.0, 1.0)
    s = jnp.where(a < eps, 0.0, s)
    t = jnp.where(a < eps, t_degenerate_d1, t)

    both_degenerate = (a < eps) & (e < eps)
    s = jnp.where(both_degenerate, 0.0, s)
    t = jnp.where(both_degenerate, 0.0, t)

    p1 = a1 + s[:, None] * d1
    p2 = a2 + t[:, None] * d2
    return jnp.linalg.norm(p1 - p2, axis=1)


def select_active_external_pairs_jax(
    z,
    q_legs,
    human_capsules,
    human_mask,
    activation_distance,
    always_keep_nearest,
    filtering_enabled,
    external_pair_slots,
):
    """Select active external pairs while keeping fixed output shapes."""
    a_robot, b_robot = capsule_endpoints_all(z, q_legs)
    h_a = human_capsules[:, :3]
    h_b = human_capsules[:, 3:6]
    h_r = human_capsules[:, 6]

    robot_idx = ALL_EXTERNAL_PAIR_INDICES[:, 0]
    human_idx = ALL_EXTERNAL_PAIR_INDICES[:, 1]

    distances = _segment_distances(
        a_robot[robot_idx], b_robot[robot_idx],
        h_a[human_idx], h_b[human_idx],
    )
    clearances = distances - RADII[robot_idx] - h_r[human_idx]
    present = human_mask[human_idx]
    sortable_clearances = jnp.where(present, clearances, jnp.inf)

    nearest_order = jnp.argsort(sortable_clearances)
    ranks = jnp.zeros_like(nearest_order).at[nearest_order].set(
        jnp.arange(nearest_order.shape[0], dtype=nearest_order.dtype)
    )

    keep_by_threshold = clearances <= activation_distance
    keep_by_rank = ranks < always_keep_nearest
    filtered_active = present & (keep_by_threshold | keep_by_rank)
    full_active = present
    active = jnp.where(filtering_enabled, filtered_active, full_active)

    selected_order = jnp.argsort(jnp.where(active, clearances, jnp.inf))
    selected_flat = selected_order[:external_pair_slots]
    selected_pairs = ALL_EXTERNAL_PAIR_INDICES[selected_flat]
    selected_mask = active[selected_flat]
    selected_clearances = jnp.where(
        selected_mask, clearances[selected_flat], jnp.inf,
    )

    return selected_pairs, selected_mask, selected_clearances
