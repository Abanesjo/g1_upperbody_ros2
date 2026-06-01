"""MarkerArray publisher for collision geometry visualization."""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA

from g1_cbf.jax_kinematics import (
    capsule_endpoints_np, BODY_NAMES, HALF_LENGTHS, RADII, N_BODIES,
    COLLISION_PAIR_INDICES, compute_sphere_counts,
)

_COLORS = {
    'torso': (0.2, 0.8, 0.2, 0.3),
    'left_arm': (0.2, 0.4, 0.9, 0.3),
    'right_arm': (0.9, 0.3, 0.2, 0.3),
    'left_shoulder': (0.3, 0.3, 0.9, 0.3),
    'right_shoulder': (0.9, 0.2, 0.3, 0.3),
    'left_thigh': (0.2, 0.7, 0.7, 0.3),
    'right_thigh': (0.7, 0.7, 0.2, 0.3),
    'left_hand': (0.55, 0.0, 0.9, 0.45),
    'right_hand': (0.55, 0.0, 0.9, 0.45),
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


def _axis_to_quat(a, b):
    """Quaternion aligning Z with the a-b axis."""
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
    return Rot.from_matrix(R).as_quat()


def _np_sphere_centers(a, b, n):
    """Interpolate n sphere centers between endpoints (numpy)."""
    if n == 1:
        return np.array([(a + b) / 2.0])
    t = np.linspace(0.0, 1.0, n)
    return (1.0 - t[:, None]) * b[None, :] + t[:, None] * a[None, :]


class ColliderVisualizer:
    """Publishes MarkerArray for collision geometry."""

    def __init__(self, node, geometry_type='capsules',
                 sphere_interpolation_level=0, sphere_radius_gain=1.0):
        self.geometry_type = str(geometry_type).lower()
        if self.geometry_type not in ('capsules', 'spheres'):
            raise ValueError(
                "collision_geometry must be 'capsules' or 'spheres'"
            )
        self.sphere_interp = sphere_interpolation_level
        self.sphere_rg = sphere_radius_gain
        self.sphere_counts = compute_sphere_counts(sphere_interpolation_level)
        self.pub = node.create_publisher(
            MarkerArray, '/robot_colliders', 10,
        )
        self.dist_pub = node.create_publisher(
            MarkerArray, '/collision_distances', 10,
        )

    def publish(self, stamp, q_controlled, q_legs=None):
        a_all, b_all, radii = capsule_endpoints_np(q_controlled, q_legs)
        half_lengths = np.asarray(HALF_LENGTHS)

        if self.geometry_type == 'spheres':
            msg = self._build_spheres(stamp, a_all, b_all, radii)
        else:
            msg = self._build_capsules(stamp, a_all, b_all, radii, half_lengths)

        self.pub.publish(msg)

    def _build_capsules(self, stamp, a_all, b_all, radii, half_lengths):
        msg = MarkerArray()
        self._delete_all_once(stamp, msg, '_colliders_initialized', 'colliders')
        mid = 0
        for i in range(N_BODIES):
            color = _COLORS.get(BODY_NAMES[i], (0.5, 0.5, 0.5, 0.3))
            radius = float(radii[i])
            diam = 2.0 * radius
            seg_half = float(half_lengths[i]) - radius
            shaft_len = 2.0 * seg_half
            center = (a_all[i] + b_all[i]) / 2.0
            quat = _axis_to_quat(a_all[i], b_all[i])

            if shaft_len <= 1e-9:
                msg.markers.append(self._make_marker(
                    stamp, mid, Marker.SPHERE, center, quat,
                    diam, diam, diam, color,
                ))
                mid += 1
                continue

            msg.markers.append(self._make_marker(
                stamp, mid, Marker.CYLINDER, center, quat,
                diam, diam, shaft_len, color,
            ))
            mid += 1
            for ep in (a_all[i], b_all[i]):
                msg.markers.append(self._make_marker(
                    stamp, mid, Marker.SPHERE, ep, quat,
                    diam, diam, diam, color,
                ))
                mid += 1

        self._cleanup(stamp, msg, mid)
        return msg

    def _build_spheres(self, stamp, a_all, b_all, radii):
        msg = MarkerArray()
        self._delete_all_once(stamp, msg, '_colliders_initialized', 'colliders')
        mid = 0
        identity_quat = [0.0, 0.0, 0.0, 1.0]
        half_lengths = np.asarray(HALF_LENGTHS)
        for i in range(N_BODIES):
            color = _COLORS.get(BODY_NAMES[i], (0.5, 0.5, 0.5, 0.3))
            r = float(radii[i])
            L = 2.0 * float(half_lengths[i])  # full capsule length, not segment
            n_base = max(1, round(L / (2.0 * r)))
            n_total = n_base + max(0, n_base - 1) * self.sphere_interp
            diam = 2.0 * r * self.sphere_rg

            if n_total == 1:
                t_values = [0.5]
            else:
                t_values = np.linspace(0.0, 1.0, n_total)

            for t in t_values:
                c = (1.0 - t) * b_all[i] + t * a_all[i]
                msg.markers.append(self._make_marker(
                    stamp, mid, Marker.SPHERE, c, identity_quat,
                    diam, diam, diam, color,
                ))
                mid += 1

        self._cleanup(stamp, msg, mid)
        return msg

    def _cleanup(self, stamp, msg, mid):
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

    def _delete_all_once(self, stamp, msg, attr_name, namespace):
        if getattr(self, attr_name, False):
            return
        m = Marker()
        m.header.frame_id = 'pelvis'
        m.header.stamp = stamp
        m.ns = namespace
        m.action = Marker.DELETEALL
        msg.markers.append(m)
        setattr(self, attr_name, True)

    def publish_distances(self, stamp, q_controlled, human_capsules=None,
                          q_legs=None, active_external_pairs=None,
                          active_internal_pairs=None):
        a_all, b_all, radii = capsule_endpoints_np(q_controlled, q_legs)

        msg = MarkerArray()
        self._delete_all_once(stamp, msg, '_distances_initialized', 'distances')
        if self.geometry_type == 'spheres':
            groups = self._dist_spheres(
                a_all, b_all, human_capsules, active_external_pairs,
                active_internal_pairs,
            )
        else:
            groups = self._dist_capsules(
                a_all, b_all, human_capsules, active_external_pairs,
            )

        idx = 0
        for pairs, color in groups:
            if not pairs:
                continue
            msg.markers.append(self._make_line_list(stamp, idx, pairs, color))
            idx += 1

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

    def _dist_capsules(self, a_all, b_all, human_capsules,
                       active_external_pairs=None):
        self_pairs = []
        human_pairs = []

        for i, j in COLLISION_PAIR_INDICES:
            p1, p2 = _closest_points_segments(
                a_all[i], b_all[i], a_all[j], b_all[j],
            )
            self_pairs.append((p1, p2))

        if human_capsules:
            if active_external_pairs is None:
                active_external_pairs = [
                    (i, h_idx)
                    for h_idx in range(len(human_capsules))
                    for i in range(N_BODIES)
                ]
            for i, h_idx in active_external_pairs:
                if i < 0 or i >= N_BODIES:
                    continue
                if h_idx < 0 or h_idx >= len(human_capsules):
                    continue
                hcap = human_capsules[h_idx]
                p1, p2 = _closest_points_segments(
                    a_all[i], b_all[i], hcap['a'], hcap['b'],
                )
                human_pairs.append((p1, p2))

        return (
            (self_pairs, (1.0, 1.0, 0.0, 0.5)),
            (human_pairs, (1.0, 0.5, 0.0, 0.5)),
        )

    def _dist_spheres(self, a_all, b_all, human_capsules,
                      active_external_pairs=None, active_internal_pairs=None):
        self_pairs = []
        human_pairs = []
        robot_centers = [
            _np_sphere_centers(a_all[i], b_all[i], self.sphere_counts[i])
            for i in range(N_BODIES)
        ]

        # Self-collision: filtered active sphere-sphere pairs when available.
        if active_internal_pairs is None:
            active_internal_pairs = [
                (i, si, j, sj)
                for i, j in COLLISION_PAIR_INDICES
                for si in range(self.sphere_counts[i])
                for sj in range(self.sphere_counts[j])
            ]
        for i, si, j, sj in active_internal_pairs:
            if i < 0 or i >= N_BODIES or j < 0 or j >= N_BODIES:
                continue
            if si < 0 or si >= len(robot_centers[i]):
                continue
            if sj < 0 or sj >= len(robot_centers[j]):
                continue
            self_pairs.append((robot_centers[i][si], robot_centers[j][sj]))

        # Human-robot: line for every robot-sphere to human-sphere pair
        if human_capsules:
            if active_external_pairs is None:
                active_external_pairs = [
                    (i, h_idx)
                    for h_idx in range(len(human_capsules))
                    for i in range(N_BODIES)
                ]
            human_centers = []
            for hcap in human_capsules:
                h_len = np.linalg.norm(hcap['a'] - hcap['b']) + 2.0 * hcap['radius']
                h_n = max(1, round(h_len / (2.0 * hcap['radius'])))
                h_n += max(0, h_n - 1) * self.sphere_interp
                human_centers.append(_np_sphere_centers(hcap['a'], hcap['b'], h_n))
            for i, h_idx in active_external_pairs:
                if i < 0 or i >= N_BODIES:
                    continue
                if h_idx < 0 or h_idx >= len(human_centers):
                    continue
                ci = robot_centers[i]
                h_centers = human_centers[h_idx]
                for si in range(self.sphere_counts[i]):
                    for sj in range(len(h_centers)):
                        human_pairs.append((ci[si], h_centers[sj]))

        return (
            (self_pairs, (1.0, 1.0, 0.0, 0.5)),
            (human_pairs, (1.0, 0.5, 0.0, 0.5)),
        )

    @staticmethod
    def _make_line_list(stamp, marker_id, point_pairs, color):
        m = Marker()
        m.header.frame_id = 'pelvis'
        m.header.stamp = stamp
        m.ns = 'distances'
        m.id = marker_id
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale = Vector3(x=0.005, y=0.0, z=0.0)
        m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        for p1, p2 in point_pairs:
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
