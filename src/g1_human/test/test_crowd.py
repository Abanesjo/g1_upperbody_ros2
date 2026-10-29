import numpy as np
import pytest

from g1_cbf.jax_kinematics import CONTROLLED_JOINTS, LEG_JOINTS
from g1_human.crowd import KNEE_SWING, STANDING_POSE, CrowdSimulator
from g1_human.human_body import PELVIS_HEIGHT, STANDING_KNEE_ANGLE

CENTER = np.array([1.0, -2.0])
DT = 1.0 / 30.0


def _crowd(seed=0, **kwargs):
    params = {'num_humans': 30, 'side_length': 10.0, 'center': CENTER}
    params.update(kwargs)
    return CrowdSimulator(rng=np.random.default_rng(seed), **params)


def test_nobody_starts_near_the_centre_and_everybody_stays_in_the_square():
    crowd = _crowd(spawn_keepout_radius=2.0)
    assert (np.linalg.norm(crowd.position - CENTER, axis=1) >= 2.0).all()

    for _ in range(3000):  # 100 s
        crowd.step(DT)
        assert (np.abs(crowd.position - CENTER) <= 5.0 + 1e-9).all()


def test_humans_walk_between_waypoints_within_their_speed_range():
    crowd = _crowd()
    start = crowd.position.copy()
    first_waypoints = crowd.waypoint.copy()
    speeds = []
    for _ in range(900):  # 30 s
        crowd.step(DT)
        speeds.append(crowd.speed.copy())

    speeds = np.array(speeds)
    assert speeds.min() >= 0.0
    assert speeds.max() <= 1.2 + 1e-9
    assert np.mean(np.linalg.norm(crowd.position - start, axis=1) > 0.5) > 0.9
    # Most have reached their first waypoint and picked another.
    assert np.mean((crowd.waypoint != first_waypoints).any(axis=1)) > 0.9


def test_the_seed_fixes_the_crowd():
    a, b = _crowd(seed=3), _crowd(seed=3)
    for _ in range(100):
        a.step(DT)
        b.step(DT)

    np.testing.assert_array_equal(a.position, b.position)
    assert not np.array_equal(
        _crowd(seed=3).position, _crowd(seed=4).position,
    )


def test_a_standing_human_holds_the_standing_pose():
    crowd = _crowd(num_humans=2)
    crowd.speed[:] = 0.0

    q, q_legs, knees, pelvis_height = crowd.walking_pose()

    np.testing.assert_allclose(q, np.tile(STANDING_POSE, (2, 1)))
    np.testing.assert_allclose(q_legs, 0.0)
    np.testing.assert_allclose(knees, STANDING_KNEE_ANGLE)
    np.testing.assert_allclose(pelvis_height, PELVIS_HEIGHT)


def test_walking_swings_the_legs_in_antiphase_and_each_arm_against_its_leg():
    crowd = _crowd(num_humans=1)
    crowd.speed[:] = 1.0
    hips = (LEG_JOINTS.index('left_hip_pitch_joint'),
            LEG_JOINTS.index('right_hip_pitch_joint'))
    shoulders = (CONTROLLED_JOINTS.index('left_shoulder_pitch_joint'),
                 CONTROLLED_JOINTS.index('right_shoulder_pitch_joint'))

    swings = 0
    for phase in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        crowd.phase[:] = phase
        q, q_legs, _, _ = crowd.walking_pose()
        assert q_legs[0, hips[0]] == pytest.approx(-q_legs[0, hips[1]])
        for hip, shoulder in zip(hips, shoulders):
            # Positive pitch is backward for both joints.
            arm = q[0, shoulder] - STANDING_POSE[shoulder]
            assert arm * q_legs[0, hip] <= 0.0
            swings += abs(arm) > 0.05
    assert swings > 12


def test_the_knee_bends_most_in_mid_swing():
    crowd = _crowd(num_humans=1)
    crowd.speed[:] = 1.0
    crowd.phase[:] = 0.0  # left leg passing under the body, moving forward

    _, _, knees, _ = crowd.walking_pose()

    assert knees[0, 0] == pytest.approx(STANDING_KNEE_ANGLE + KNEE_SWING)
    assert knees[0, 1] == pytest.approx(STANDING_KNEE_ANGLE)


@pytest.mark.parametrize('kwargs', [
    {'num_humans': 0},
    {'side_length': 0.0},
    {'min_speed': 1.5},
    {'spawn_keepout_radius': 5.0},
])
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _crowd(**kwargs)
