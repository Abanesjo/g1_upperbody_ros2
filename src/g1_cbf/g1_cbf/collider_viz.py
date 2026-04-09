"""MarkerArray publisher for capsule collision visualization."""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA

from g1_cbf.jax_kinematics import (
    capsule_endpoints_np, BODY_NAMES, HALF_LENGTHS, RADII, N_BODIES,
    COLLISION_PAIR_INDICES,
)

_COLORS = {
    'torso': (0.2, 0.8, 0.2, 0.3),
    'left_arm': (0.2, 0.4, 0.9, 0.3),
    'right_arm': (0.9, 0.3, 0.2, 0.3),
    'left_shoulder': (0.3, 0.3, 0.9, 0.3),
    'right_shoulder': (0.9, 0.2, 0.3, 0.3),
    'left_thigh': (0.2, 0.7, 0.7, 0.3),
    'right_thigh': (0.7, 0.7, 0.2, 0.3),
}


def _closest_points_segments(a1, b1, a2, b2):
    """Closest points between two line segments (numpy, no JAX)."""
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
            if abs(denom) > eps:
                s = np.clip((b_val * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            t = (b_val * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b_val - c) / a, 0.0, 1.0)

    return a1 + s * d1, a2 + t * d2


class ColliderVisualizer:
    """Publishes capsule MarkerArray for collision geometry."""

    def __init__(self, node):
        self.pub = node.create_publisher(
            MarkerArray, '/robot_colliders', 10,
        )
        self.dist_pub = node.create_publisher(
            MarkerArray, '/collision_distances', 10,
        )

    def publish(self, stamp, q_controlled, q_legs=None):
        """Publish capsule markers for current joint configuration.

        Args:
            stamp: ROS2 time stamp.
            q_controlled: (8,) numpy array of controlled joint positions.
            q_legs: (6,) numpy array of leg joint positions, or None.
        """
        a_all, b_all, radii = capsule_endpoints_np(q_controlled, q_legs)
        half_lengths = np.asarray(HALF_LENGTHS)

        msg = MarkerArray()
        mid = 0

        for i in range(N_BODIES):
            name = BODY_NAMES[i]
            radius = float(radii[i])
            seg_half = float(half_lengths[i]) - radius
            shaft_len = 2.0 * seg_half
            diam = 2.0 * radius
            color = _COLORS.get(name, (0.5, 0.5, 0.5, 0.3))

            a = a_all[i]
            b = b_all[i]
            center = (a + b) / 2.0
            axis = a - b
            length = np.linalg.norm(axis)

            if length > 1e-6:
                z_ax = axis / length
                up = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(z_ax, up)) > 0.999:
                    up = np.array([1.0, 0.0, 0.0])
                x_ax = np.cross(up, z_ax)
                x_ax /= np.linalg.norm(x_ax)
                y_ax = np.cross(z_ax, x_ax)
                R = np.column_stack([x_ax, y_ax, z_ax])
            else:
                R = np.eye(3)

            quat = Rot.from_matrix(R).as_quat()

            # Cylinder shaft
            m = self._make_marker(
                stamp, mid, Marker.CYLINDER,
                center, quat, diam, diam, shaft_len, color,
            )
            msg.markers.append(m)
            mid += 1

            # Sphere caps
            for endpoint in (a, b):
                m = self._make_marker(
                    stamp, mid, Marker.SPHERE,
                    endpoint, quat, diam, diam, diam, color,
                )
                msg.markers.append(m)
                mid += 1

        # Clean stale markers
        prev = getattr(self, '_prev_n_markers', 0)
        for j in range(mid, prev):
            m = Marker()
            m.header.frame_id = 'pelvis'
            m.header.stamp = stamp
            m.ns = 'colliders'
            m.id = j
            m.action = Marker.DELETE
            msg.markers.append(m)
        self._prev_n_markers = mid

        self.pub.publish(msg)

    def publish_distances(self, stamp, q_controlled, human_capsules=None,
                          q_legs=None):
        """Publish line segments between closest points on collision pairs.

        Args:
            stamp: ROS2 time stamp.
            q_controlled: (8,) numpy array of controlled joint positions.
            human_capsules: optional list of {'a': (3,), 'b': (3,)} dicts.
            q_legs: (6,) numpy array of leg joint positions, or None.
        """
        a_all, b_all, radii = capsule_endpoints_np(q_controlled, q_legs)

        msg = MarkerArray()
        idx = 0

        # Self-collision pairs (yellow)
        for i, j in COLLISION_PAIR_INDICES:
            p1, p2 = _closest_points_segments(
                a_all[i], b_all[i], a_all[j], b_all[j],
            )
            msg.markers.append(self._make_line(
                stamp, idx, p1, p2, (1.0, 1.0, 0.0, 1.0),
            ))
            idx += 1

        # Human-robot pairs (orange)
        if human_capsules:
            for hcap in human_capsules:
                for i in range(N_BODIES):
                    p1, p2 = _closest_points_segments(
                        a_all[i], b_all[i], hcap['a'], hcap['b'],
                    )
                    msg.markers.append(self._make_line(
                        stamp, idx, p1, p2, (1.0, 0.5, 0.0, 1.0),
                    ))
                    idx += 1

        # Clean stale
        prev = getattr(self, '_prev_n_distances', 0)
        for k in range(idx, prev):
            m = Marker()
            m.header.frame_id = 'pelvis'
            m.header.stamp = stamp
            m.ns = 'distances'
            m.id = k
            m.action = Marker.DELETE
            msg.markers.append(m)
        self._prev_n_distances = idx

        self.dist_pub.publish(msg)

    @staticmethod
    def _make_line(stamp, marker_id, p1, p2, color):
        m = Marker()
        m.header.frame_id = 'pelvis'
        m.header.stamp = stamp
        m.ns = 'distances'
        m.id = marker_id
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale = Vector3(x=0.005, y=0.0, z=0.0)
        m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        m.points.append(Point(
            x=float(p1[0]), y=float(p1[1]), z=float(p1[2]),
        ))
        m.points.append(Point(
            x=float(p2[0]), y=float(p2[1]), z=float(p2[2]),
        ))
        return m

    @staticmethod
    def _make_marker(stamp, marker_id, marker_type,
                     center, quat, sx, sy, sz, color):
        m = Marker()
        m.header.frame_id = 'pelvis'
        m.header.stamp = stamp
        m.ns = 'colliders'
        m.id = marker_id
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.position = Point(
            x=float(center[0]),
            y=float(center[1]),
            z=float(center[2]),
        )
        m.pose.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]),
            z=float(quat[2]), w=float(quat[3]),
        )
        m.scale = Vector3(
            x=float(sx), y=float(sy), z=float(sz),
        )
        r, g, b, a = color
        m.color = ColorRGBA(r=r, g=g, b=b, a=a)
        return m
