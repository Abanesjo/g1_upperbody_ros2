import numpy as np
import pytest

from g1_cbf.collider_viz import _closest_points_segments


def _closest_points_one_pair(a1, b1, a2, b2):
    """The per-pair version _closest_points_segments replaced."""
    d1 = b1 - a1
    d2 = b2 - a2
    r = a1 - a2
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    eps = 1e-8
    if a < eps and e < eps:
        return a1.copy(), a2.copy()
    if a < eps:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e < eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b_val = np.dot(d1, d2)
            denom = a * e - b_val * b_val
            s = np.clip((b_val * f - c * e) / denom, 0.0, 1.0) if abs(denom) > eps else 0.0
            t = (b_val * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b_val - c) / a, 0.0, 1.0)
    return a1 + s * d1, a2 + t * d2


def test_vectorized_closest_points_match_the_per_pair_version():
    rng = np.random.default_rng(0)
    a1, b1, a2, b2 = rng.normal(size=(4, 200, 3))
    b1[:10] = a1[:10]                      # first segment is a point
    b2[10:20] = a2[10:20]                  # second segment is a point
    b1[20:25] = a1[20:25]
    b2[20:25] = a2[20:25]                  # both are points
    b2[25:35] = a2[25:35] + (b1 - a1)[25:35]  # parallel segments

    p1, p2 = _closest_points_segments(a1, b1, a2, b2)

    for k in range(len(a1)):
        q1, q2 = _closest_points_one_pair(a1[k], b1[k], a2[k], b2[k])
        np.testing.assert_allclose(p1[k], q1, atol=1e-9)
        np.testing.assert_allclose(p2[k], q2, atol=1e-9)


def test_closest_points_of_crossing_segments():
    p1, p2 = _closest_points_segments(
        np.array([[-1.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.5, -1.0, 2.0]]), np.array([[0.5, 1.0, 2.0]]),
    )

    assert p1[0] == pytest.approx([0.5, 0.0, 0.0])
    assert p2[0] == pytest.approx([0.5, 0.0, 2.0])
