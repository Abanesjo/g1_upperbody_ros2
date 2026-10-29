import numpy as np
import pytest

from g1_cbf.jax_kinematics import (
    BODY_NAMES,
    CONTROLLED_JOINT_DEFAULTS,
    HALF_LENGTHS,
    RADII,
    capsule_endpoints_np,
)
from g1_human.human_body import (
    HUMAN_CAPSULE_NAMES,
    STANDING_KNEE_ANGLE,
    human_capsules,
)


def _previous_construction(a_all, b_all, radius_scale):
    """The per-capsule construction g1_human_node used before human_capsules."""
    thighs = {
        BODY_NAMES.index('left_thigh'): 'left',
        BODY_NAMES.index('right_thigh'): 'right',
    }
    shin_radius = 0.065 * radius_scale
    shin_seg_half = 0.15 * 1.5 - shin_radius
    bend = np.radians(10.0)
    capsules = []
    thigh_bottoms = {}
    for i in range(len(BODY_NAMES)):
        r = float(RADII[i]) * radius_scale
        hl = float(HALF_LENGTHS[i]) * (1.5 if i in thighs else 1.0)
        center = (a_all[i] + b_all[i]) / 2.0
        axis = a_all[i] - b_all[i]
        direction = axis / np.linalg.norm(axis)
        seg_half = max(0.0, hl - r)
        a = center + seg_half * direction
        b = center - seg_half * direction
        if i in thighs:
            thigh_bottoms[thighs[i]] = b
        capsules.append((a, b, r, hl))
    for side in ('left', 'right'):
        top = thigh_bottoms[side]
        axis = a_all[BODY_NAMES.index(f'{side}_thigh')] - top
        direction = axis / np.linalg.norm(axis)
        lateral = np.cross(direction, [0.0, 0.0, 1.0])
        lateral /= np.linalg.norm(lateral)
        c, s = np.cos(bend), np.sin(bend)
        shin_dir = (
            direction * c
            + np.cross(lateral, direction) * s
            + lateral * np.dot(lateral, direction) * (1 - c)
        )
        capsules.append((
            top, top - 2.0 * max(0.0, shin_seg_half) * shin_dir,
            shin_radius, 0.225,
        ))
    return capsules


@pytest.mark.parametrize('radius_scale', [1.0, 1.3])
@pytest.mark.parametrize('seed', [0, 1])
def test_standing_human_matches_the_previous_construction(radius_scale, seed):
    q = CONTROLLED_JOINT_DEFAULTS + np.random.default_rng(seed).uniform(
        -0.4, 0.4, CONTROLLED_JOINT_DEFAULTS.shape,
    )
    a_all, b_all, _ = capsule_endpoints_np(q)

    a, b, radii, half_lengths = human_capsules(
        a_all, b_all, (STANDING_KNEE_ANGLE, STANDING_KNEE_ANGLE), radius_scale,
    )

    previous = _previous_construction(a_all, b_all, radius_scale)
    assert len(HUMAN_CAPSULE_NAMES) == len(previous) == 11
    for k, (prev_a, prev_b, prev_r, prev_hl) in enumerate(previous):
        np.testing.assert_allclose(a[k], prev_a, atol=1e-12)
        np.testing.assert_allclose(b[k], prev_b, atol=1e-12)
        assert radii[k] == pytest.approx(prev_r)
        assert half_lengths[k] == pytest.approx(prev_hl)


def test_humans_in_a_batch_are_independent():
    rng = np.random.default_rng(2)
    qs = CONTROLLED_JOINT_DEFAULTS + rng.uniform(-0.4, 0.4, (3, 11))
    bodies = [capsule_endpoints_np(q) for q in qs]
    a_all = np.stack([a for a, _, _ in bodies])
    b_all = np.stack([b for _, b, _ in bodies])
    knees = rng.uniform(0.0, 1.0, (3, 2))

    a, b, _, _ = human_capsules(a_all, b_all, knees)

    for i in range(3):
        a_i, b_i, _, _ = human_capsules(a_all[i], b_all[i], knees[i])
        np.testing.assert_allclose(a[i], a_i)
        np.testing.assert_allclose(b[i], b_i)


def test_knee_flexion_swings_the_shin_back_about_the_knee():
    a_all, b_all, _ = capsule_endpoints_np(CONTROLLED_JOINT_DEFAULTS)
    straight_a, straight_b, _, _ = human_capsules(a_all, b_all, (0.0, 0.0))
    bent_a, bent_b, _, _ = human_capsules(a_all, b_all, (0.8, 0.8))

    for shin in (HUMAN_CAPSULE_NAMES.index('left_shin'),
                 HUMAN_CAPSULE_NAMES.index('right_shin')):
        # The shin still hangs from the knee, with the same length...
        np.testing.assert_allclose(bent_a[shin], straight_a[shin])
        assert np.linalg.norm(bent_b[shin] - bent_a[shin]) == pytest.approx(
            np.linalg.norm(straight_b[shin] - straight_a[shin])
        )
        # ...and its lower end moves back (-x) and up.
        assert bent_b[shin][0] < straight_b[shin][0] - 0.1
        assert bent_b[shin][2] > straight_b[shin][2]
