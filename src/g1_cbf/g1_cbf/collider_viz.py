"""MarkerArray publisher for collision geometry visualization."""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA

from g1_cbf.jax_kinematics import (
    capsule_endpoints_np, BODY_NAMES, HALF_LENGTHS, RADII, N_BODIES,
    COLLISION_PAIR_INDICES, HEAD_COLLIDER_BODY_INDEX, compute_sphere_counts,
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
_HEAD_COLLIDER_COLOR = (1.0, 1.0, 0.0, 0.45)


def _closest_points_segments(a1, b1, a2, b2):
    """Closest points between pairs of line segments, one pair per row.

    Vectorized numpy: with a crowd there are hundreds of pairs per tick, and
    their count changes every tick, which a jitted JAX version would have to
    pad away.
    """
    d1 = b1 - a1
    d2 = b2 - a2
    r = a1 - a2
    a = np.sum(d1 * d1, axis=-1)
    e = np.sum(d2 * d2, axis=-1)
    f = np.sum(d2 * r, axis=-1)
    c = np.sum(d1 * r, axis=-1)
    b_val = np.sum(d1 * d2, axis=-1)

    eps = 1e-8
    safe_a = np.where(a < eps, 1.0, a)
    safe_e = np.where(e < eps, 1.0, e)
    denom = a * e - b_val * b_val
    parallel = np.abs(denom) <= eps
    s = np.where(
        parallel, 0.0,
        np.clip((b_val * f - c * e) / np.where(parallel, 1.0, denom), 0.0, 1.0),
    )
    t = (b_val * s + f) / safe_e
    s = np.where(t < 0.0, np.clip(-c / safe_a, 0.0, 1.0), s)
    s = np.where(t > 1.0, np.clip((b_val - c) / safe_a, 0.0, 1.0), s)
    t = np.clip(t, 0.0, 1.0)
    # Second segment is a point: project it on the first.
    s = np.where(e < eps, np.clip(-c / safe_a, 0.0, 1.0), s)
    t = np.where(e < eps, 0.0, t)
    # First segment is a point: project it on the second.
    t = np.where(a < eps, np.clip(f / safe_e, 0.0, 1.0), t)
    s = np.where(a < eps, 0.0, s)
    return a1 + s[..., None] * d1, a2 + t[..., None] * d2


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
                 sphere_interpolation_level=0, sphere_radius_gain=1.0,
                 head_collider_radius=0.3):
        self.geometry_type = str(geometry_type).lower()
        if self.geometry_type not in ('capsules', 'spheres'):
            raise ValueError(
                "collision_geometry must be 'capsules' or 'spheres'"
            )
        self.sphere_interp = sphere_interpolation_level
        self.sphere_rg = sphere_radius_gain
        self.head_collider_radius = float(head_collider_radius)
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

        mid = self._append_head_collider(stamp, msg, mid, a_all)
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

        mid = self._append_head_collider(stamp, msg, mid, a_all)
        self._cleanup(stamp, msg, mid)
        return msg

    def _append_head_collider(self, stamp, msg, mid, a_all):
        center = a_all[HEAD_COLLIDER_BODY_INDEX]
        diam = 2.0 * self.head_collider_radius
        msg.markers.append(self._make_marker(
            stamp, mid, Marker.SPHERE, center, [0.0, 0.0, 0.0, 1.0],
            diam, diam, diam, _HEAD_COLLIDER_COLOR,
        ))
        return mid + 1

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
        body_i = [i for i, _ in COLLISION_PAIR_INDICES]
        body_j = [j for _, j in COLLISION_PAIR_INDICES]
        self_pairs = list(zip(*_closest_points_segments(
            a_all[body_i], b_all[body_i], a_all[body_j], b_all[body_j],
        )))

        human_pairs = []
        if human_capsules:
            if active_external_pairs is None:
                active_external_pairs = [
                    (i, h_idx)
                    for h_idx in range(len(human_capsules))
                    for i in range(N_BODIES)
                ]
            pairs = [
                (i, h_idx) for i, h_idx in active_external_pairs
                if 0 <= i < N_BODIES and 0 <= h_idx < len(human_capsules)
            ]
            if pairs:
                robot = [i for i, _ in pairs]
                human_pairs = list(zip(*_closest_points_segments(
                    a_all[robot], b_all[robot],
                    np.array([human_capsules[h]['a'] for _, h in pairs]),
                    np.array([human_capsules[h]['b'] for _, h in pairs]),
                )))

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
        coordinates = np.asarray(point_pairs, dtype=np.float64).reshape(-1, 3)
        m.points = [
            Point(x=x, y=y, z=z) for x, y, z in coordinates.tolist()
        ]
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
