"""MarkerArray publisher for collision geometry visualization.

Publishes /robot_colliders with all bodies as capsules or boxes,
exactly matching the CBF geometry.
"""

from scipy.spatial.transform import Rotation as Rot

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA

_COLORS = {
    'torso': (0.2, 0.8, 0.2, 0.3),
    'left_arm': (0.2, 0.4, 0.9, 0.3),
    'right_arm': (0.9, 0.3, 0.2, 0.3),
    'left_thigh': (0.2, 0.7, 0.7, 0.3),
    'right_thigh': (0.7, 0.7, 0.2, 0.3),
}


class ColliderVisualizer:
    """Publishes MarkerArray for collision geometry."""

    def __init__(self, node, kinematics, geometry_type='capsules'):
        self.kin = kinematics
        self.geometry_type = geometry_type
        self.pub = node.create_publisher(
            MarkerArray, '/robot_colliders', 10,
        )
        self.dist_pub = node.create_publisher(
            MarkerArray, '/collision_distances', 10,
        )

    def publish(self, stamp):
        if self.geometry_type == 'boxes':
            self._publish_boxes(stamp)
        elif self.geometry_type == 'spheres':
            pass  # spheres use publish_spheres() with precomputed data
        else:
            self._publish_capsules(stamp)

    def _publish_capsules(self, stamp):
        msg = MarkerArray()
        mid = 0

        for name, body in self.kin.collision_bodies.items():
            radius = body['radius']
            half_len = body['half_length']
            seg_half = half_len - radius
            shaft_len = 2.0 * seg_half
            color = _COLORS.get(name, (0.5, 0.5, 0.5, 0.3))

            center, rot = self.kin.get_collision_pose(name)
            quat = Rot.from_matrix(rot).as_quat()
            axis = rot[:, 2]

            diam = 2.0 * radius
            m = self._make_marker(
                stamp, mid, Marker.CYLINDER,
                center, quat, diam, diam, shaft_len, color,
            )
            msg.markers.append(m)
            mid += 1

            for sign in (+1, -1):
                sph_c = center + sign * seg_half * axis
                m = self._make_marker(
                    stamp, mid, Marker.SPHERE,
                    sph_c, quat, diam, diam, diam, color,
                )
                msg.markers.append(m)
                mid += 1

        self.pub.publish(msg)

    def _publish_boxes(self, stamp):
        msg = MarkerArray()
        mid = 0

        for name, body in self.kin.collision_bodies.items():
            radius = body['radius']
            half_len = body['half_length']
            color = _COLORS.get(name, (0.5, 0.5, 0.5, 0.3))

            center, rot = self.kin.get_collision_pose(name)
            quat = Rot.from_matrix(rot).as_quat()

            # Box: X,Y = 2*radius, Z = 2*half_length
            m = self._make_marker(
                stamp, mid, Marker.CUBE,
                center, quat,
                2.0 * radius, 2.0 * radius, 2.0 * half_len,
                color,
            )
            msg.markers.append(m)
            mid += 1

        self.pub.publish(msg)

    @staticmethod
    def _make_marker(
        stamp, marker_id, marker_type,
        center, quat, sx, sy, sz, color,
    ):
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

    def publish_spheres(self, stamp, sphere_data):
        """Publish sphere decomposition as SPHERE markers.

        Parameters
        ----------
        sphere_data : dict
            Maps body_name -> {'centers': (n,3), 'radius': float, 'n_spheres': int}
        """
        msg = MarkerArray()
        mid = 0
        identity_quat = [0.0, 0.0, 0.0, 1.0]

        for name, data in sphere_data.items():
            color = _COLORS.get(name, (0.5, 0.5, 0.5, 0.3))
            diam = 2.0 * data['radius']

            for i in range(data['n_spheres']):
                center = data['centers'][i]
                m = self._make_marker(
                    stamp, mid, Marker.SPHERE,
                    center, identity_quat,
                    diam, diam, diam, color,
                )
                msg.markers.append(m)
                mid += 1

        # Delete stale markers from previous frames
        prev = getattr(self, '_prev_n_sphere_markers', 0)
        for j in range(mid, prev):
            m = Marker()
            m.header.frame_id = 'pelvis'
            m.header.stamp = stamp
            m.ns = 'colliders'
            m.id = j
            m.action = Marker.DELETE
            msg.markers.append(m)
        self._prev_n_sphere_markers = mid

        self.pub.publish(msg)

    def publish_distances(self, stamp, closest_points):
        """Publish line segments between closest points."""
        msg = MarkerArray()
        n = len(closest_points)
        for i, (p1, p2) in enumerate(closest_points):
            m = Marker()
            m.header.frame_id = 'pelvis'
            m.header.stamp = stamp
            m.ns = 'distances'
            m.id = i
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale = Vector3(x=0.005, y=0.0, z=0.0)
            m.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
            m.points.append(Point(
                x=float(p1[0]), y=float(p1[1]), z=float(p1[2]),
            ))
            m.points.append(Point(
                x=float(p2[0]), y=float(p2[1]), z=float(p2[2]),
            ))
            msg.markers.append(m)

        # Delete stale markers from previous frames
        prev = getattr(self, '_prev_n_distances', 0)
        for j in range(n, prev):
            m = Marker()
            m.header.frame_id = 'pelvis'
            m.header.stamp = stamp
            m.ns = 'distances'
            m.id = j
            m.action = Marker.DELETE
            msg.markers.append(m)
        self._prev_n_distances = n

        self.dist_pub.publish(msg)
