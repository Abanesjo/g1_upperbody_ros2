"""MarkerArray publisher for capsule collision visualization."""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA

from g1_cbf.jax_kinematics import (
    capsule_endpoints_np, BODY_NAMES, HALF_LENGTHS, RADII, N_BODIES,
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


class ColliderVisualizer:
    """Publishes capsule MarkerArray for collision geometry."""

    def __init__(self, node):
        self.pub = node.create_publisher(
            MarkerArray, '/robot_colliders', 10,
        )

    def publish(self, stamp, q_controlled):
        """Publish capsule markers for current joint configuration.

        Args:
            stamp: ROS2 time stamp.
            q_controlled: (8,) numpy array of controlled joint positions.
        """
        a_all, b_all, radii = capsule_endpoints_np(q_controlled)
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
