"""JAX broadphase selection for active collision pairs."""

import jax.numpy as jnp

from g1_cbf.jax_kinematics import (
    BODY_INDEX,
    COLLISION_PAIR_INDICES,
    N_HUMAN_CAPSULES,
    RADII,
    capsule_endpoints_all,
    sphere_centers,
)


EXTERNAL_ROBOT_BODY_NAMES = (
    'torso',
    'left_arm',
    'right_arm',
    'left_shoulder',
    'right_shoulder',
    'left_hand',
    'right_hand',
)
EXTERNAL_ROBOT_BODY_INDICES = tuple(
    BODY_INDEX[name] for name in EXTERNAL_ROBOT_BODY_NAMES
)
N_EXTERNAL_ROBOT_BODIES = len(EXTERNAL_ROBOT_BODY_INDICES)

ALL_EXTERNAL_PAIR_INDICES = jnp.array(
    [
        (robot_idx, human_idx)
        for human_idx in range(N_HUMAN_CAPSULES)
        for robot_idx in EXTERNAL_ROBOT_BODY_INDICES
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


def make_internal_sphere_pair_indices(sphere_counts):
    """Build all internal robot sphere-pair candidates for sphere mode."""
    pairs = []
    for body_a, body_b in COLLISION_PAIR_INDICES:
        for sphere_a in range(sphere_counts[body_a]):
            for sphere_b in range(sphere_counts[body_b]):
                pairs.append((body_a, sphere_a, body_b, sphere_b))
    return jnp.array(pairs, dtype=jnp.int32)


def _padded_sphere_centers(a_all, b_all, sphere_counts, max_count):
    centers = []
    for i, count in enumerate(sphere_counts):
        c = sphere_centers(a_all[i], b_all[i], count)
        if count < max_count:
            pad = jnp.zeros((max_count - count, 3), dtype=c.dtype)
            c = jnp.concatenate([c, pad], axis=0)
        centers.append(c)
    return jnp.stack(centers)


def select_active_internal_sphere_pairs_jax(
    z,
    q_legs,
    activation_distance,
    always_keep_nearest,
    filtering_enabled,
    all_pair_indices,
    sphere_counts,
    max_robot_spheres,
    sphere_radius_gain,
    internal_pair_slots,
):
    """Select active internal sphere pairs while keeping fixed output shapes."""
    a_robot, b_robot = capsule_endpoints_all(z, q_legs)
    robot_centers = _padded_sphere_centers(
        a_robot, b_robot, sphere_counts, max_robot_spheres,
    )

    body_a = all_pair_indices[:, 0]
    sphere_a = all_pair_indices[:, 1]
    body_b = all_pair_indices[:, 2]
    sphere_b = all_pair_indices[:, 3]

    ca = robot_centers[body_a, sphere_a]
    cb = robot_centers[body_b, sphere_b]
    clearances = (
        jnp.linalg.norm(ca - cb, axis=1)
        - RADII[body_a] * sphere_radius_gain
        - RADII[body_b] * sphere_radius_gain
    )

    nearest_order = jnp.argsort(clearances)
    ranks = jnp.zeros_like(nearest_order).at[nearest_order].set(
        jnp.arange(nearest_order.shape[0], dtype=nearest_order.dtype)
    )

    keep_by_threshold = clearances <= activation_distance
    keep_by_rank = ranks < always_keep_nearest
    filtered_active = keep_by_threshold | keep_by_rank
    active = jnp.where(filtering_enabled, filtered_active, jnp.ones_like(filtered_active))

    selected_order = jnp.argsort(jnp.where(active, clearances, jnp.inf))
    selected_flat = selected_order[:internal_pair_slots]
    selected_pairs = all_pair_indices[selected_flat]
    selected_mask = active[selected_flat]
    selected_clearances = jnp.where(
        selected_mask, clearances[selected_flat], jnp.inf,
    )

    return selected_pairs, selected_mask, selected_clearances


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
