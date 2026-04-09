#!/usr/bin/env python3
"""Human capsule collider publisher.

Uses the same JAX FK as the CBF node to compute capsule endpoints,
applies a fixed pelvis-to-pelvis transform, and publishes all collision
capsules as a CapsuleArray on /human/colliders.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as Rot
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion, Vector3
from std_msgs.msg import ColorRGBA

from g1_cbf.jax_kinematics import (
    capsule_endpoints_np, BODY_NAMES, HALF_LENGTHS, RADII,
    N_BODIES, CONTROLLED_JOINTS,
)
from g1_cbf_msg.msg import Capsule, CapsuleArray

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class G1HumanNode(Node):
    def __init__(self):
        super().__init__('g1_human_node')

        # Parameters
        self.declare_parameter('human_x', 0.6)
        self.declare_parameter('human_y', 0.0)
        self.declare_parameter('human_z', 0.0)
        self.declare_parameter('human_roll', 0.0)
        self.declare_parameter('human_pitch', 0.0)
        self.declare_parameter('human_yaw', 3.14159)
        self.declare_parameter('rate', 50.0)

        # Joint state: 8 controlled joints, default neutral (zeros)
        self.q_controlled = np.zeros(8)
        self.q_des = None

        # Publishers
        self.capsule_pub = self.create_publisher(
            CapsuleArray, '/human/colliders', SENSOR_QOS,
        )
        self.viz_pub = self.create_publisher(
            MarkerArray, '/human/collider_markers', 10,
        )

        # Subscriber for human joint commands
        self.create_subscription(
            JointState, '/human/joint_commands',
            self._joint_cmd_cb, SENSOR_QOS,
        )

        # Timer
        rate = self.get_parameter('rate').value
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'g1_human_node ready — publishing at {rate:.0f} Hz'
        )

    def _get_pelvis_transform(self):
        """Build rotation + translation from robot pelvis to human pelvis."""
        tx = self.get_parameter('human_x').value
        ty = self.get_parameter('human_y').value
        tz = self.get_parameter('human_z').value
        roll = self.get_parameter('human_roll').value
        pitch = self.get_parameter('human_pitch').value
        yaw = self.get_parameter('human_yaw').value

        R = Rot.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        t = np.array([tx, ty, tz])
        return R, t

    def _joint_cmd_cb(self, msg: JointState):
        """Accept joint commands — extract the 8 controlled joints."""
        name_to_pos = dict(zip(msg.name, msg.position))
        q = np.zeros(8)
        for i, jname in enumerate(CONTROLLED_JOINTS):
            if jname in name_to_pos:
                q[i] = name_to_pos[jname]
        self.q_des = q

    def _tick(self):
        q = self.q_des if self.q_des is not None else np.zeros(8)

        # FK via JAX (returns numpy)
        a_all, b_all, radii = capsule_endpoints_np(q)
        half_lengths = np.asarray(HALF_LENGTHS)

        R_base, t_base = self._get_pelvis_transform()

        # Build capsule message + viz data
        capsule_msg = CapsuleArray()
        stamp = self.get_clock().now().to_msg()
        capsule_msg.header.stamp = stamp
        capsule_msg.header.frame_id = 'pelvis'

        capsule_data = []
        for i in range(N_BODIES):
            a_world = R_base @ a_all[i] + t_base
            b_world = R_base @ b_all[i] + t_base

            capsule = Capsule()
            capsule.a = Point(
                x=float(a_world[0]),
                y=float(a_world[1]),
                z=float(a_world[2]),
            )
            capsule.b = Point(
                x=float(b_world[0]),
                y=float(b_world[1]),
                z=float(b_world[2]),
            )
            capsule.radius = float(radii[i])
            capsule.name = BODY_NAMES[i]
            capsule_msg.capsules.append(capsule)
            capsule_data.append((
                BODY_NAMES[i], float(radii[i]),
                float(half_lengths[i]), a_world, b_world,
            ))

        # Visualization (capsules only)
        viz_msg = self._viz_capsules(stamp, capsule_data)

        self.capsule_pub.publish(capsule_msg)
        self.viz_pub.publish(viz_msg)

    _COLOR = (0.9, 0.5, 0.1, 0.3)  # orange for human

    def _viz_capsules(self, stamp, capsule_data):
        msg = MarkerArray()
        mid = 0
        for name, radius, half_length, a_world, b_world in capsule_data:
            diam = 2.0 * radius
            seg_half = half_length - radius
            shaft_len = 2.0 * seg_half
            center = (a_world + b_world) / 2.0
            quat = self._axis_to_quat(a_world, b_world)

            msg.markers.append(self._make_marker(
                stamp, mid, Marker.CYLINDER, center, quat,
                diam, diam, shaft_len,
            ))
            mid += 1
            for endpoint in (a_world, b_world):
                msg.markers.append(self._make_marker(
                    stamp, mid, Marker.SPHERE, endpoint,
                    [0, 0, 0, 1], diam, diam, diam,
                ))
                mid += 1

        # Clean stale markers
        prev = getattr(self, '_prev_n_markers', 0)
        for j in range(mid, prev):
            m = Marker()
            m.header.frame_id = 'pelvis'
            m.header.stamp = stamp
            m.ns = 'human_colliders'
            m.id = j
            m.action = Marker.DELETE
            msg.markers.append(m)
        self._prev_n_markers = mid
        return msg

    def _axis_to_quat(self, a_world, b_world):
        axis = a_world - b_world
        length = np.linalg.norm(axis)
        if length > 1e-6:
            z_ax = axis / length
            up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(z_ax, up)) > 0.999:
                up = np.array([1.0, 0.0, 0.0])
            x_ax = np.cross(up, z_ax)
            x_ax /= np.linalg.norm(x_ax)
            y_ax = np.cross(z_ax, x_ax)
            R_viz = np.column_stack([x_ax, y_ax, z_ax])
        else:
            R_viz = np.eye(3)
        return Rot.from_matrix(R_viz).as_quat()

    def _make_marker(self, stamp, mid, marker_type, center, quat, sx, sy, sz):
        m = Marker()
        m.header.frame_id = 'pelvis'
        m.header.stamp = stamp
        m.ns = 'human_colliders'
        m.id = mid
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.position = Point(
            x=float(center[0]), y=float(center[1]), z=float(center[2]),
        )
        m.pose.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]),
            z=float(quat[2]), w=float(quat[3]),
        )
        m.scale = Vector3(x=float(sx), y=float(sy), z=float(sz))
        r, g, b, a = self._COLOR
        m.color = ColorRGBA(r=r, g=g, b=b, a=a)
        return m


def main(args=None):
    rclpy.init(args=args)
    node = G1HumanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
