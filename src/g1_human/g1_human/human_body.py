"""Collision capsules of a simulated human, built on the G1 collider model.

A simulated human is the G1 collision model of g1_cbf.jax_kinematics with
1.5x longer thighs, radii scaled by human_radius_scale and two shins added
below the thighs: 11 capsules in the order of HUMAN_CAPSULE_NAMES. Functions
are vectorized over any leading (per-human) dimensions.
"""

import numpy as np

from g1_cbf.jax_kinematics import BODY_NAMES, HALF_LENGTHS, RADII

HUMAN_CAPSULE_NAMES = tuple(BODY_NAMES) + ('left_shin', 'right_shin')
THIGH_SCALE = 1.5
SHIN_RADIUS = 0.065
SHIN_HALF_LENGTH = 0.15 * THIGH_SCALE
STANDING_KNEE_ANGLE = np.radians(10.0)
PELVIS_HEIGHT = 0.784202

_THIGHS = (BODY_NAMES.index('left_thigh'), BODY_NAMES.index('right_thigh'))


def human_capsules(a_body, b_body, knee_angles, radius_scale=1.0):
    """Return the human's capsules in its pelvis frame.

    Args:
        a_body, b_body: (..., N_BODIES, 3) G1 capsule endpoints from
            capsule_endpoints_all, one set per human.
        knee_angles: (..., 2) left and right knee flexion [rad]; the shin
            leaves the bottom of the thigh rotated backward by this angle
            about the pelvis y axis.
        radius_scale: factor on every radius.

    Returns:
        a, b: (..., 11, 3) endpoints, ordered as HUMAN_CAPSULE_NAMES.
        radii, half_lengths: (11,).
    """
    a_body = np.asarray(a_body, dtype=np.float64)
    b_body = np.asarray(b_body, dtype=np.float64)
    knee_angles = np.asarray(knee_angles, dtype=np.float64)
    radii = np.asarray(RADII, dtype=np.float64) * radius_scale
    half_lengths = np.array(HALF_LENGTHS, dtype=np.float64)
    half_lengths[list(_THIGHS)] *= THIGH_SCALE

    # Keep each capsule's centre and axis, and fit the straight segment to the
    # (lengthened, rescaled) capsule.
    center = 0.5 * (a_body + b_body)
    direction = _unit(a_body - b_body)
    seg_half = np.maximum(0.0, half_lengths - radii)[:, None]
    a = center + seg_half * direction
    b = center - seg_half * direction

    shin_radius = SHIN_RADIUS * radius_scale
    shin_length = 2.0 * max(0.0, SHIN_HALF_LENGTH - shin_radius)
    shin_a = []
    shin_b = []
    for side, thigh in enumerate(_THIGHS):
        knee = b[..., thigh, :]
        # Up the thigh, from the knee to the top endpoint of the G1 thigh.
        up = _unit(a_body[..., thigh, :] - knee)
        shin_a.append(knee)
        shin_b.append(
            knee - shin_length * _rotate_about_y(up, knee_angles[..., side])
        )

    return (
        np.concatenate([a, np.stack(shin_a, axis=-2)], axis=-2),
        np.concatenate([b, np.stack(shin_b, axis=-2)], axis=-2),
        np.concatenate([radii, [shin_radius, shin_radius]]),
        np.concatenate([half_lengths, [SHIN_HALF_LENGTH, SHIN_HALF_LENGTH]]),
    )


def _unit(vectors):
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe = np.where(norm > 1e-6, norm, 1.0)
    return np.where(norm > 1e-6, vectors / safe, np.array([0.0, 0.0, 1.0]))


def _rotate_about_y(vectors, angles):
    c = np.cos(angles)[..., None]
    s = np.sin(angles)[..., None]
    x = vectors[..., 0:1]
    y = vectors[..., 1:2]
    z = vectors[..., 2:3]
    return np.concatenate([x * c + z * s, y, z * c - x * s], axis=-1)
