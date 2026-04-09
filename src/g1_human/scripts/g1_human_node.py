#!/usr/bin/env python3
"""Human capsule collider publisher.

Uses the same JAX FK as the CBF node to compute capsule endpoints,
applies a fixed pelvis-to-pelvis transform, and publishes all collision
capsules as a CapsuleArray on /human/colliders.
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

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
        self.declare_parameter('publish_viz', False)
        self.declare_parameter('collision_geometry', 'capsules')
        self.declare_parameter('sphere_interpolation_level', 0)
        self.declare_parameter('sphere_radius_gain', 1.0)

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

        # Thigh indices for extension
        THIGH_INDICES = {
            BODY_NAMES.index('left_thigh'): 'left',
            BODY_NAMES.index('right_thigh'): 'right',
        }
        THIGH_SCALE = 1.5
        shin_radius = 0.065
        shin_half_length = 0.15 * 1.5
        shin_seg_half = shin_half_length - shin_radius
        shin_knee_bend = np.radians(10.0)  # slight forward bend at knee

        capsule_data = []
        thigh_bottoms = {}  # side -> bottom endpoint in local frame (for shin attachment)

        for i in range(N_BODIES):
            r = float(radii[i])
            hl = float(half_lengths[i])
            a_local = a_all[i]
            b_local = b_all[i]

            # Extend thigh capsules for the human
            if i in THIGH_INDICES:
                hl_ext = hl * THIGH_SCALE
                seg_half_ext = hl_ext - r
                center = (a_local + b_local) / 2.0
                axis = a_local - b_local
                length = np.linalg.norm(axis)
                if length > 1e-6:
                    direction = axis / length
                else:
                    direction = np.array([0.0, 0.0, 1.0])
                a_local = center + seg_half_ext * direction
                b_local = center - seg_half_ext * direction
                hl = hl_ext
                thigh_bottoms[THIGH_INDICES[i]] = b_local.copy()

            a_world = R_base @ a_local + t_base
            b_world = R_base @ b_local + t_base

            capsule = Capsule()
            capsule.a = Point(x=float(a_world[0]), y=float(a_world[1]), z=float(a_world[2]))
            capsule.b = Point(x=float(b_world[0]), y=float(b_world[1]), z=float(b_world[2]))
            capsule.radius = r
            capsule.name = BODY_NAMES[i]
            capsule_msg.capsules.append(capsule)
            capsule_data.append((BODY_NAMES[i], r, hl, a_world, b_world))

        # Shin capsules — start at bottom of extended thigh, continue in same direction
        for side in ('left', 'right'):
            thigh_idx = BODY_NAMES.index(f'{side}_thigh')
            a_thigh = a_all[thigh_idx]
            b_thigh = thigh_bottoms[side]
            axis = a_thigh - b_thigh
            length = np.linalg.norm(axis)
            direction = axis / length if length > 1e-6 else np.array([0.0, 0.0, 1.0])

            # Rotate shin direction forward (Y axis bend) for knee angle
            # Find a perpendicular axis to rotate around (lateral axis)
            up = np.array([0.0, 0.0, 1.0])
            lateral = np.cross(direction, up)
            if np.linalg.norm(lateral) < 1e-6:
                lateral = np.array([0.0, 1.0, 0.0])
            lateral = lateral / np.linalg.norm(lateral)
            # Rodrigues rotation of direction around lateral axis
            c, s = np.cos(shin_knee_bend), np.sin(shin_knee_bend)
            shin_dir = direction * c + np.cross(lateral, direction) * s + lateral * np.dot(lateral, direction) * (1 - c)

            # Shin top = thigh bottom, shin extends in bent direction
            shin_top = b_thigh
            shin_bottom = shin_top - 2.0 * shin_seg_half * shin_dir
            a_world = R_base @ shin_top + t_base
            b_world = R_base @ shin_bottom + t_base

            capsule = Capsule()
            capsule.a = Point(x=float(a_world[0]), y=float(a_world[1]), z=float(a_world[2]))
            capsule.b = Point(x=float(b_world[0]), y=float(b_world[1]), z=float(b_world[2]))
            capsule.radius = shin_radius
            capsule.name = f'{side}_shin'
            capsule_msg.capsules.append(capsule)
            capsule_data.append((f'{side}_shin', shin_radius, shin_half_length, a_world, b_world))

        self.capsule_pub.publish(capsule_msg)

        if self.get_parameter('publish_viz').value:
            geom = self.get_parameter('collision_geometry').value
            if geom == 'spheres':
                viz_msg = self._viz_spheres(stamp, capsule_data)
            elif geom == 'boxes':
                viz_msg = self._viz_boxes(stamp, capsule_data)
            else:
                viz_msg = self._viz_capsules(stamp, capsule_data)
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

        self._cleanup_stale(stamp, msg, mid)
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

    def _cleanup_stale(self, stamp, msg, mid):
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

    def _viz_boxes(self, stamp, capsule_data):
        msg = MarkerArray()
        mid = 0
        for name, radius, half_length, a_world, b_world in capsule_data:
            center = (a_world + b_world) / 2.0
            quat = self._axis_to_quat(a_world, b_world)
            msg.markers.append(self._make_marker(
                stamp, mid, Marker.CUBE, center, quat,
                2.0 * radius, 2.0 * radius, 2.0 * half_length,
            ))
            mid += 1
        self._cleanup_stale(stamp, msg, mid)
        return msg

    def _viz_spheres(self, stamp, capsule_data):
        interp = self.get_parameter('sphere_interpolation_level').value
        rg = self.get_parameter('sphere_radius_gain').value
        msg = MarkerArray()
        mid = 0
        identity_quat = [0.0, 0.0, 0.0, 1.0]

        for name, radius, half_length, a_world, b_world in capsule_data:
            L = 2.0 * half_length  # full capsule length, not segment
            n_base = max(1, round(L / (2.0 * radius)))
            n_total = n_base + max(0, n_base - 1) * interp
            diam = 2.0 * radius * rg

            if n_total == 1:
                t_values = [0.5]
            else:
                t_values = np.linspace(0.0, 1.0, n_total)

            for t in t_values:
                c = (1.0 - t) * b_world + t * a_world
                msg.markers.append(self._make_marker(
                    stamp, mid, Marker.SPHERE, c,
                    identity_quat, diam, diam, diam,
                ))
                mid += 1

        self._cleanup_stale(stamp, msg, mid)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = G1HumanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
