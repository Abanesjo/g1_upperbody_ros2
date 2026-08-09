import numpy as np
import pytest

from g1_cbf.active_pairs import (
    EXTERNAL_ROBOT_BODY_INDICES,
    _segment_distances,
)
from g1_cbf.jax_kinematics import (
    COLLISION_PAIR_INDICES,
    CONTROLLED_JOINT_DEFAULTS,
    N_HUMAN_CAPSULES,
    N_LEG_JOINTS,
    RADII,
    capsule_endpoints_np,
)
from g1_logging.geometry import (
    EXTERNAL_PAIR_LABELS,
    INTERNAL_PAIR_LABELS,
    capsule_surface_clearance,
    closest_segment_distance,
    compute_external_clearances,
    compute_internal_clearances,
)


@pytest.mark.parametrize(
    ('a1', 'b1', 'a2', 'b2', 'expected'),
    [
        # Parallel unit segments.
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), 1.0),
        # Segments cross at their midpoints.
        ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), 0.0),
        # Both line segments degenerate to points.
        ((0, 0, 0), (0, 0, 0), (1, 2, 2), (1, 2, 2), 3.0),
        # One degenerate segment projects onto the other.
        ((0.5, 2, 0), (0.5, 2, 0), (0, 0, 0), (1, 0, 0), 2.0),
    ],
)
def test_closest_segment_distance_known_cases(a1, b1, a2, b2, expected):
    assert closest_segment_distance(a1, b1, a2, b2) == pytest.approx(
        expected, abs=1e-9,
    )


def test_closest_segment_distance_matches_cbf_broadphase():
    rng = np.random.default_rng(42)
    a1 = rng.normal(size=(32, 3))
    b1 = rng.normal(size=(32, 3))
    a2 = rng.normal(size=(32, 3))
    b2 = rng.normal(size=(32, 3))

    # Include the degeneracies handled explicitly by the CBF broadphase.
    b1[0] = a1[0]
    b2[1] = a2[1]
    b1[2] = a1[2]
    b2[2] = a2[2]

    cbf_distances = np.asarray(_segment_distances(a1, b1, a2, b2))
    logging_distances = np.array([
        closest_segment_distance(x1, y1, x2, y2)
        for x1, y1, x2, y2 in zip(a1, b1, a2, b2)
    ])

    np.testing.assert_allclose(
        logging_distances, cbf_distances, rtol=2e-5, atol=2e-5,
    )


def test_capsule_surface_clearance_is_signed():
    clearance = capsule_surface_clearance(
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        0.1,
        0.2,
    )

    assert clearance == pytest.approx(-0.3)


def test_internal_clearances_match_authoritative_cbf_geometry():
    q = CONTROLLED_JOINT_DEFAULTS.copy()
    q_legs = np.zeros(N_LEG_JOINTS)

    actual = compute_internal_clearances(q, q_legs)
    a_all, b_all, radii = capsule_endpoints_np(q, q_legs)

    expected = np.array([
        capsule_surface_clearance(
            a_all[body_a],
            b_all[body_a],
            a_all[body_b],
            b_all[body_b],
            radii[body_a],
            radii[body_b],
        )
        for body_a, body_b in COLLISION_PAIR_INDICES
    ])

    assert actual.shape == (21,)
    assert len(INTERNAL_PAIR_LABELS) == 21
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def _quat_rotation_matrix_xyzw(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm([x, y, z, w])
    x, y, z, w = np.array([x, y, z, w]) / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ])


def test_external_clearances_match_cbf_pair_order_and_world_transform():
    q = CONTROLLED_JOINT_DEFAULTS.copy()
    q_legs = np.zeros(N_LEG_JOINTS)
    translation = np.array([1.0, -2.0, 0.5])
    quaternion = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])

    human_a = np.zeros((N_HUMAN_CAPSULES, 3))
    human_b = np.zeros((N_HUMAN_CAPSULES, 3))
    human_radii = np.full(N_HUMAN_CAPSULES, 0.08)
    human_present = np.zeros(N_HUMAN_CAPSULES, dtype=bool)
    human_present[[0, 3]] = True
    human_a[0], human_b[0] = (1.2, -2.1, 0.4), (1.2, -2.1, 1.0)
    human_a[3], human_b[3] = (0.7, -1.8, 0.1), (1.1, -1.4, 0.4)

    actual = compute_external_clearances(
        q,
        q_legs,
        translation,
        quaternion,
        human_a,
        human_b,
        human_radii,
        human_present,
    )

    pelvis_a, pelvis_b, radii = capsule_endpoints_np(q, q_legs)
    rotation = _quat_rotation_matrix_xyzw(quaternion)
    world_a = pelvis_a @ rotation.T + translation
    world_b = pelvis_b @ rotation.T + translation
    expected = np.full((N_HUMAN_CAPSULES, 7), np.nan)
    for human_index in (0, 3):
        for external_index, robot_index in enumerate(
            EXTERNAL_ROBOT_BODY_INDICES
        ):
            expected[human_index, external_index] = (
                capsule_surface_clearance(
                    world_a[robot_index],
                    world_b[robot_index],
                    human_a[human_index],
                    human_b[human_index],
                    float(RADII[robot_index]),
                    human_radii[human_index],
                )
            )

    assert actual.shape == (11, 7)
    assert len(EXTERNAL_PAIR_LABELS) == 77
    assert np.isnan(actual[~human_present]).all()
    np.testing.assert_allclose(
        actual[human_present],
        expected[human_present],
        rtol=1e-10,
        atol=1e-10,
    )
