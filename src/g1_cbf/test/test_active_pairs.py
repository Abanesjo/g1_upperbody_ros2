import jax.numpy as jnp
import numpy as np

from g1_cbf.active_pairs import (
    ALL_EXTERNAL_PAIR_INDICES,
    EXTERNAL_ROBOT_BODY_INDICES,
    RADII,
    _segment_distances,
    select_active_external_pairs_jax,
)
from g1_cbf.jax_kinematics import (
    CONTROLLED_JOINT_DEFAULTS,
    N_HUMAN_CAPSULES,
    N_LEG_JOINTS,
    capsule_endpoints_all,
)


SLOTS = 40


def _humans(present, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(
        [-1.5, -1.5, -0.5], [1.5, 1.5, 0.8], (N_HUMAN_CAPSULES, 3),
    )
    capsules = np.column_stack([
        centers + [0.0, 0.0, 0.1],
        centers - [0.0, 0.0, 0.1],
        np.full(N_HUMAN_CAPSULES, 0.05),
    ])
    mask = np.zeros(N_HUMAN_CAPSULES, dtype=bool)
    mask[:present] = True
    return jnp.array(capsules), jnp.array(mask)


def _select(capsules, mask, filtering, activation=0.35, keep_nearest=4):
    """Selector output and the sorted clearances of all present pairs."""
    z = jnp.array(CONTROLLED_JOINT_DEFAULTS)
    q_legs = jnp.zeros(N_LEG_JOINTS)
    pairs, selected, clearances = select_active_external_pairs_jax(
        z, q_legs, capsules, mask,
        jnp.array(activation), jnp.array(keep_nearest, dtype=jnp.int32),
        jnp.array(filtering), SLOTS,
    )
    a, b = capsule_endpoints_all(z, q_legs)
    robot = ALL_EXTERNAL_PAIR_INDICES[:, 0]
    human = ALL_EXTERNAL_PAIR_INDICES[:, 1]
    all_clearances = np.asarray(
        _segment_distances(
            a[robot], b[robot], capsules[human, :3], capsules[human, 3:6],
        )
        - RADII[robot] - capsules[human, 6]
    )
    present = np.asarray(mask)[np.asarray(human)]
    return (
        np.asarray(pairs), np.asarray(selected), np.asarray(clearances),
        np.sort(all_clearances[present]),
    )


def test_without_the_filter_the_nearest_present_pairs_fill_the_slots():
    # 20 present capsules are 140 pairs for 40 slots.
    pairs, selected, clearances, nearest = _select(
        *_humans(20), filtering=False,
    )

    assert selected.all()
    np.testing.assert_allclose(clearances, nearest[:SLOTS])
    assert (pairs[:, 1] < 20).all()


def test_without_the_filter_every_present_pair_is_kept_when_they_fit():
    pairs, selected, clearances, _ = _select(*_humans(3), filtering=False)

    assert sorted(map(tuple, pairs[selected].tolist())) == sorted(
        (robot, human)
        for robot in EXTERNAL_ROBOT_BODY_INDICES for human in range(3)
    )
    assert np.isinf(clearances[~selected]).all()


def test_the_filter_keeps_near_pairs_and_the_nearest_few():
    activation = 0.3
    _, selected, clearances, nearest = _select(
        *_humans(20), filtering=True, activation=activation, keep_nearest=4,
    )

    expected = min(
        SLOTS, max(4, int(np.count_nonzero(nearest <= activation))),
    )
    assert selected.sum() == expected
    np.testing.assert_allclose(clearances[:expected], nearest[:expected])
