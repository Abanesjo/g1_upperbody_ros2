#!/usr/bin/env python3
"""Rate-limited CBF collision visualizer.

Runs outside the CBF control node so marker construction and RViz publishing
cannot block the safety-filter timer.
"""

import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

from builtin_interfaces.msg import Time
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

from g1_cbf.collider_viz import ColliderVisualizer
from g1_cbf.jax_kinematics import CONTROLLED_JOINTS, LEG_JOINTS
from g1_cbf_msg.msg import ActiveCollisionPairs, CapsuleArray

PELVIS_HEIGHT = 0.784202
WORLD_FRAME = 'world'
PELVIS_FRAME = 'pelvis'


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class G1CBFVizNode(Node):
    def __init__(self):
        super().__init__('g1_cbf_viz_node')

        self.declare_parameter('publish_viz', False)
        self.declare_parameter('viz_rate', 5.0)
        self.declare_parameter('collision_geometry', 'capsules')
        self.declare_parameter('sphere_interpolation_level', 0)
        self.declare_parameter('sphere_radius_gain', 1.0)
        self.declare_parameter('head_collider_radius', 0.3)

        self.q_ctrl = None
        self.q_legs = np.zeros(len(LEG_JOINTS))
        self._human_capsules = []
        self._human_capsules_frame = PELVIS_FRAME
        self._pelvis_position = None
        self._pelvis_quat = None
        self._active_external_pairs = None
        self._active_internal_pairs = None

        self.viz = ColliderVisualizer(
            self,
            geometry_type=self.get_parameter('collision_geometry').value,
            sphere_interpolation_level=(
                self.get_parameter('sphere_interpolation_level').value
            ),
            sphere_radius_gain=self.get_parameter('sphere_radius_gain').value,
            head_collider_radius=(
                self.get_parameter('head_collider_radius').value
            ),
        )

        self.create_subscription(
            JointState, '/joint_states', self._joint_states_cb, SENSOR_QOS,
        )
        self.create_subscription(
            CapsuleArray, '/human/colliders', self._human_cb, SENSOR_QOS,
        )
        self.create_subscription(
            PoseStamped, '/pose/pelvis', self._pelvis_pose_cb, SENSOR_QOS,
        )
        self.create_subscription(
            ActiveCollisionPairs, '/cbf/active_collision_pairs',
            self._active_pairs_cb, SENSOR_QOS,
        )

        rate = float(self.get_parameter('viz_rate').value)
        if rate <= 0.0:
            self.get_logger().warn('viz_rate must be positive; using 5 Hz')
            rate = 5.0
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'g1_cbf_viz_node ready - publish_viz='
            f'{self.get_parameter("publish_viz").value}, rate={rate:.1f} Hz'
        )

    def _joint_states_cb(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))

        q = np.zeros(len(CONTROLLED_JOINTS))
        for i, jname in enumerate(CONTROLLED_JOINTS):
            if jname in name_to_pos:
                q[i] = name_to_pos[jname]
        self.q_ctrl = q

        ql = np.zeros(len(LEG_JOINTS))
        for i, jname in enumerate(LEG_JOINTS):
            if jname in name_to_pos:
                ql[i] = name_to_pos[jname]
        self.q_legs = ql

    def _human_cb(self, msg: CapsuleArray):
        capsules = []
        for c in msg.capsules:
            capsules.append({
                'a': np.array([c.a.x, c.a.y, c.a.z]),
                'b': np.array([c.b.x, c.b.y, c.b.z]),
                'radius': float(c.radius),
            })
        self._human_capsules = capsules
        self._human_capsules_frame = self._normalize_frame(msg.header.frame_id)

    def _pelvis_pose_cb(self, msg: PoseStamped):
        self._pelvis_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=np.float64)
        self._pelvis_quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ], dtype=np.float64)

    def _active_pairs_cb(self, msg: ActiveCollisionPairs):
        external_pairs = []
        count = min(
            len(msg.robot_body_index),
            len(msg.human_capsule_index),
        )
        for i in range(count):
            external_pairs.append((
                int(msg.robot_body_index[i]),
                int(msg.human_capsule_index[i]),
            ))
        self._active_external_pairs = external_pairs

        internal_pairs = []
        internal_count = min(
            len(msg.internal_body_a_index),
            len(msg.internal_sphere_a_index),
            len(msg.internal_body_b_index),
            len(msg.internal_sphere_b_index),
        )
        for i in range(internal_count):
            internal_pairs.append((
                int(msg.internal_body_a_index[i]),
                int(msg.internal_sphere_a_index[i]),
                int(msg.internal_body_b_index[i]),
                int(msg.internal_sphere_b_index[i]),
            ))
        self._active_internal_pairs = internal_pairs

    def _tick(self):
        if not self.get_parameter('publish_viz').value:
            return
        if self.q_ctrl is None:
            return

        stamp = Time()
        human_capsules = self._human_capsules_for_viz()
        self.viz.publish(stamp, self.q_ctrl, self.q_legs)
        self.viz.publish_distances(
            stamp, self.q_ctrl, human_capsules or None, self.q_legs,
            self._active_external_pairs, self._active_internal_pairs,
        )

    def _human_capsules_for_viz(self):
        capsules = []
        for capsule in self._human_capsules:
            a, b = self._capsule_endpoints_in_pelvis(capsule)
            capsules.append({
                'a': a,
                'b': b,
                'radius': capsule['radius'],
            })
        return capsules

    def _capsule_endpoints_in_pelvis(self, capsule):
        frame = self._human_capsules_frame
        a = capsule['a']
        b = capsule['b']
        if frame == WORLD_FRAME:
            return (
                self._world_to_pelvis(a),
                self._world_to_pelvis(b),
            )
        if frame not in ('', PELVIS_FRAME):
            self.get_logger().warn(
                f"Unsupported /human/colliders frame '{frame}'; "
                "treating capsules as pelvis-frame coordinates",
                throttle_duration_sec=2.0,
            )
        return a, b

    def _world_to_pelvis(self, point_world):
        pelvis_position, pelvis_quat = self._pelvis_pose_or_default()
        return self._quat_rotate_np(
            self._quat_conjugate_np(pelvis_quat),
            point_world - pelvis_position,
        )

    def _pelvis_pose_or_default(self):
        if self._pelvis_position is None or self._pelvis_quat is None:
            self.get_logger().warn(
                '/human/colliders is in world frame, but /pose/pelvis has '
                'not been received; using nominal pelvis pose for this tick',
                throttle_duration_sec=2.0,
            )
            return (
                np.array([0.0, 0.0, PELVIS_HEIGHT], dtype=np.float64),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            )
        return self._pelvis_position, self._normalize_quat(self._pelvis_quat)

    @staticmethod
    def _normalize_frame(frame_id):
        frame = (frame_id or PELVIS_FRAME).strip()
        return frame[1:] if frame.startswith('/') else frame

    @staticmethod
    def _normalize_quat(q):
        q = np.asarray(q, dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q / norm

    @staticmethod
    def _quat_conjugate_np(q):
        q = G1CBFVizNode._normalize_quat(q)
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

    @staticmethod
    def _quat_rotate_np(q, v):
        q = G1CBFVizNode._normalize_quat(q)
        xyz = q[:3]
        w = q[3]
        t = 2.0 * np.cross(xyz, v)
        return v + w * t + np.cross(xyz, t)


def main(args=None):
    rclpy.init(args=args)
    node = G1CBFVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
