"""Capsule collision body for CBF.

Capsule = line segment + radius. The long axis is the Z column
of the collision body rotation matrix.
"""

import numpy as np


def _skew(v):
    """Skew-symmetric matrix [v]_x."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


class Capsule3D:
    """Capsule: line segment [c - l*v, c + l*v] with radius r.

    Parameters
    ----------
    center : (3,) capsule center
    R : (3,3) rotation matrix (long axis = R[:,2])
    half_length : total half-length (segment + cap)
    radius : capsule radius
    """

    def __init__(self, center, R, half_length, radius):
        self.radius = float(radius)
        self.seg_half_len = float(half_length - radius)
        self.center = np.asarray(center, dtype=float)
        self.direction = np.asarray(R, dtype=float)[:, 2]

    def update(self, center, R):
        self.center = np.asarray(center, dtype=float)
        self.direction = np.asarray(R, dtype=float)[:, 2]

    @property
    def endpoint_a(self):
        return self.center + self.seg_half_len * self.direction

    @property
    def endpoint_b(self):
        return self.center - self.seg_half_len * self.direction
