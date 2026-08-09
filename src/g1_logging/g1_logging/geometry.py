"""Offline collision-geometry reconstruction for :mod:`g1_logging`.

The logger intentionally records state, not derived collision distances.  This
module reuses the exact robot capsule definitions and pair ordering used by
``g1_cbf`` and provides NumPy helpers for reconstructing signed capsule surface
clearance:

    closest center-line distance - radius_a - radius_b

Negative values therefore mean that the two capsule volumes overlap.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

# Offline reconstruction must not compete with the real-time CBF process for
# GPU memory. These are read when g1_cbf imports JAX below.
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'true'

import jax
import jax.numpy as jnp
import numpy as np

from g1_cbf.active_pairs import (
    EXTERNAL_ROBOT_BODY_INDICES,
    EXTERNAL_ROBOT_BODY_NAMES,
)
from g1_cbf.jax_kinematics import (
    BODY_NAMES,
    COLLISION_PAIR_INDICES,
    COLLISION_PAIRS,
    N_HUMAN_CAPSULES,
    RADII,
    capsule_endpoints_all,
)


_CAPSULE_ENDPOINTS_JIT = jax.jit(capsule_endpoints_all)
_ROBOT_RADII = np.asarray(RADII, dtype=np.float64)

INTERNAL_PAIR_LABELS: Tuple[str, ...] = tuple(
    f'{body_a} \u2194 {body_b}' for body_a, body_b in COLLISION_PAIRS
)
EXTERNAL_PAIR_LABELS: Tuple[str, ...] = tuple(
    f'{body_name} \u2194 human[{human_index:02d}]'
    for human_index in range(N_HUMAN_CAPSULES)
    for body_name in EXTERNAL_ROBOT_BODY_NAMES
)


def closest_segment_distances(
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Return closest distances for broadcast-compatible 3-D line segments.

    The implementation mirrors ``g1_cbf.active_pairs._segment_distances`` but
    runs in NumPy for offline analysis and also handles point-like segments.
    Inputs may be shape ``(3,)`` or ``(..., 3)``.
    """

    a1_array, b1_array, a2_array, b2_array = np.broadcast_arrays(
        _points(a1, 'a1'),
        _points(b1, 'b1'),
        _points(a2, 'a2'),
        _points(b2, 'b2'),
    )

    d1 = b1_array - a1_array
    d2 = b2_array - a2_array
    relative = a1_array - a2_array

    len1_sq = np.sum(d1 * d1, axis=-1)
    len2_sq = np.sum(d2 * d2, axis=-1)
    d2_relative = np.sum(d2 * relative, axis=-1)
    d1_relative = np.sum(d1 * relative, axis=-1)
    d1_d2 = np.sum(d1 * d2, axis=-1)

    eps = 1.0e-8
    safe_len1_sq = np.maximum(len1_sq, eps)
    safe_len2_sq = np.maximum(len2_sq, eps)
    denominator = len1_sq * len2_sq - d1_d2 * d1_d2

    general_s = np.zeros_like(denominator, dtype=np.float64)
    nonparallel = np.abs(denominator) > eps
    np.divide(
        d1_d2 * d2_relative - d1_relative * len2_sq,
        denominator,
        out=general_s,
        where=nonparallel,
    )
    general_s = np.clip(general_s, 0.0, 1.0)
    general_t = (d1_d2 * general_s + d2_relative) / safe_len2_sq

    s_below = np.clip(-d1_relative / safe_len1_sq, 0.0, 1.0)
    s_above = np.clip(
        (d1_d2 - d1_relative) / safe_len1_sq,
        0.0,
        1.0,
    )
    general_s = np.where(general_t < 0.0, s_below, general_s)
    general_s = np.where(general_t > 1.0, s_above, general_s)
    general_t = np.clip(general_t, 0.0, 1.0)

    s_if_second_point = np.clip(
        -d1_relative / safe_len1_sq,
        0.0,
        1.0,
    )
    segment_2_degenerate = len2_sq < eps
    s = np.where(segment_2_degenerate, s_if_second_point, general_s)
    t = np.where(segment_2_degenerate, 0.0, general_t)

    t_if_first_point = np.clip(
        d2_relative / safe_len2_sq,
        0.0,
        1.0,
    )
    segment_1_degenerate = len1_sq < eps
    s = np.where(segment_1_degenerate, 0.0, s)
    t = np.where(segment_1_degenerate, t_if_first_point, t)

    both_degenerate = segment_1_degenerate & segment_2_degenerate
    s = np.where(both_degenerate, 0.0, s)
    t = np.where(both_degenerate, 0.0, t)

    closest_1 = a1_array + s[..., None] * d1
    closest_2 = a2_array + t[..., None] * d2
    return np.linalg.norm(closest_1 - closest_2, axis=-1)


def closest_segment_distance(
    a1: Sequence[float],
    b1: Sequence[float],
    a2: Sequence[float],
    b2: Sequence[float],
) -> float:
    """Scalar convenience wrapper around :func:`closest_segment_distances`."""

    return float(closest_segment_distances(a1, b1, a2, b2))


def capsule_surface_clearance(
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
    radius_1: np.ndarray,
    radius_2: np.ndarray,
) -> np.ndarray:
    """Return signed surface clearance for broadcast-compatible capsules."""

    return (
        closest_segment_distances(a1, b1, a2, b2)
        - np.asarray(radius_1, dtype=np.float64)
        - np.asarray(radius_2, dtype=np.float64)
    )


def compute_robot_capsules(
    q_controlled: Sequence[float],
    q_legs: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute all nine robot capsules in the pelvis frame."""

    q = _finite_vector(q_controlled, 11, 'q_controlled')
    legs = _finite_vector(q_legs, 6, 'q_legs')
    endpoints_a, endpoints_b = _CAPSULE_ENDPOINTS_JIT(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(legs, dtype=jnp.float64),
    )
    return (
        np.asarray(endpoints_a, dtype=np.float64),
        np.asarray(endpoints_b, dtype=np.float64),
        _ROBOT_RADII.copy(),
    )


def compute_internal_clearances(
    q_controlled: Sequence[float],
    q_legs: Sequence[float],
) -> np.ndarray:
    """Return the 21 configured self-collision capsule clearances."""

    endpoints_a, endpoints_b, radii = compute_robot_capsules(
        q_controlled,
        q_legs,
    )
    indices_a = np.asarray(
        [pair[0] for pair in COLLISION_PAIR_INDICES],
        dtype=np.int64,
    )
    indices_b = np.asarray(
        [pair[1] for pair in COLLISION_PAIR_INDICES],
        dtype=np.int64,
    )
    return capsule_surface_clearance(
        endpoints_a[indices_a],
        endpoints_b[indices_a],
        endpoints_a[indices_b],
        endpoints_b[indices_b],
        radii[indices_a],
        radii[indices_b],
    )


def compute_external_clearances(
    q_controlled: Sequence[float],
    q_legs: Sequence[float],
    tf_translation: Sequence[float],
    tf_quaternion: Sequence[float],
    human_a: np.ndarray,
    human_b: np.ndarray,
    human_radii: Sequence[float],
    human_present: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """Return world-frame robot/human clearances with shape ``(11, 7)``.

    Rows are fixed human slots and columns follow
    ``EXTERNAL_ROBOT_BODY_NAMES``. Missing human slots are ``NaN``. This is the
    same human-major ordering as
    ``g1_cbf.active_pairs.ALL_EXTERNAL_PAIR_INDICES``.
    """

    robot_a, robot_b, robot_radii = compute_robot_capsules(
        q_controlled,
        q_legs,
    )
    robot_a_world = transform_points(
        robot_a,
        tf_translation,
        tf_quaternion,
    )
    robot_b_world = transform_points(
        robot_b,
        tf_translation,
        tf_quaternion,
    )

    human_a_array = _capsule_points(human_a, 'human_a')
    human_b_array = _capsule_points(human_b, 'human_b')
    if human_a_array.shape != human_b_array.shape:
        raise ValueError('human_a and human_b must have matching shapes')
    if human_a_array.shape[0] > N_HUMAN_CAPSULES:
        human_a_array = human_a_array[:N_HUMAN_CAPSULES]
        human_b_array = human_b_array[:N_HUMAN_CAPSULES]

    count = human_a_array.shape[0]
    human_radius_array = np.asarray(human_radii, dtype=np.float64).reshape(-1)
    if human_radius_array.size < count:
        raise ValueError('human_radii has fewer entries than human endpoints')
    human_radius_array = human_radius_array[:count]

    if human_present is None:
        present = np.ones(count, dtype=bool)
    else:
        present_values = np.asarray(human_present, dtype=bool).reshape(-1)
        if present_values.size < count:
            raise ValueError(
                'human_present has fewer entries than human endpoints'
            )
        present = present_values[:count]

    output = np.full(
        (N_HUMAN_CAPSULES, len(EXTERNAL_ROBOT_BODY_INDICES)),
        np.nan,
        dtype=np.float64,
    )
    if count == 0:
        return output

    valid_human = (
        present
        & np.all(np.isfinite(human_a_array), axis=1)
        & np.all(np.isfinite(human_b_array), axis=1)
        & np.isfinite(human_radius_array)
        & (human_radius_array > 0.0)
    )
    if not np.any(valid_human):
        return output

    robot_indices = np.asarray(
        EXTERNAL_ROBOT_BODY_INDICES,
        dtype=np.int64,
    )
    clearances = capsule_surface_clearance(
        robot_a_world[robot_indices][None, :, :],
        robot_b_world[robot_indices][None, :, :],
        human_a_array[:, None, :],
        human_b_array[:, None, :],
        robot_radii[robot_indices][None, :],
        human_radius_array[:, None],
    )
    output[:count] = np.where(valid_human[:, None], clearances, np.nan)
    return output


def transform_points(
    points: np.ndarray,
    translation: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> np.ndarray:
    """Apply an ``x,y,z,w`` quaternion and translation to 3-D points."""

    points_array = _points(points, 'points')
    offset = _finite_vector(translation, 3, 'translation')
    quaternion = _finite_vector(quaternion_xyzw, 4, 'quaternion_xyzw')
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-9:
        raise ValueError('quaternion_xyzw must be nonzero')
    quaternion = quaternion / norm

    xyz = quaternion[:3]
    twice_cross = 2.0 * np.cross(xyz, points_array)
    rotated = (
        points_array
        + quaternion[3] * twice_cross
        + np.cross(xyz, twice_cross)
    )
    return rotated + offset


def _points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f'{name} must have final dimension 3')
    return array


def _capsule_points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f'{name} must have shape (N, 3)')
    return array


def _finite_vector(
    value: Sequence[float],
    expected_size: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != expected_size:
        raise ValueError(f'{name} must have {expected_size} entries')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain only finite values')
    return array


__all__ = [
    'BODY_NAMES',
    'EXTERNAL_PAIR_LABELS',
    'EXTERNAL_ROBOT_BODY_NAMES',
    'INTERNAL_PAIR_LABELS',
    'capsule_surface_clearance',
    'closest_segment_distance',
    'closest_segment_distances',
    'compute_external_clearances',
    'compute_internal_clearances',
    'compute_robot_capsules',
    'transform_points',
]
