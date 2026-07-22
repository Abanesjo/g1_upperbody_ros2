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
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from tf2_ros import TransformException

from g1_cbf.collider_viz import ColliderVisualizer
from g1_cbf.human_capsules import (
    human_data_is_fresh,
    human_data_time,
    transform_capsule_array,
)
from g1_cbf.jax_kinematics import (
    CONTROLLED_JOINTS,
    LEG_JOINTS,
    N_HUMAN_CAPSULES,
)
from g1_cbf.tf_pose import (
    TfPoseLookup,
    normalize_frame,
    resolve_lookup_timeout_sec,
)
from g1_cbf_msg.msg import ActiveCollisionPairs, CapsuleArray

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
        self.declare_parameter('world_frame', WORLD_FRAME)
        self.declare_parameter('pelvis_frame', PELVIS_FRAME)
        self.declare_parameter('human_timeout_sec', 0.5)
        self.declare_parameter('tf_lookup_timeout_sec', 0.0)
        self.declare_parameter('tf_timeout_sec', 0.0)

        self.q_ctrl = None
        self.q_legs = np.zeros(len(LEG_JOINTS))
        self._human_capsules = []
        self._last_human_time = None
        self._world_frame = normalize_frame(
            self.get_parameter('world_frame').value,
            WORLD_FRAME,
        )
        self._pelvis_frame = normalize_frame(
            self.get_parameter('pelvis_frame').value,
            PELVIS_FRAME,
        )
        self._human_timeout_sec = float(
            self.get_parameter('human_timeout_sec').value
        )
        if self._human_timeout_sec < 0.0:
            raise ValueError('human_timeout_sec must be non-negative')
        self._tf_pose_lookup = TfPoseLookup(
            self,
            self._world_frame,
            self._pelvis_frame,
            resolve_lookup_timeout_sec(self),
        )
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
        try:
            msg_world = transform_capsule_array(
                msg,
                self._world_frame,
                self._tf_pose_lookup.buffer,
            )
        except (TransformException, ValueError) as exc:
            self.get_logger().warn(
                f"Invalid /human/colliders transform from "
                f"'{msg.header.frame_id}' to '{self._world_frame}': {exc}; "
                'keeping the last valid capsules',
                throttle_duration_sec=2.0,
            )
            return

        capsules = []
        for c in msg_world.capsules[:N_HUMAN_CAPSULES]:
            capsules.append({
                'a': np.array([c.a.x, c.a.y, c.a.z]),
                'b': np.array([c.b.x, c.b.y, c.b.z]),
                'radius': float(c.radius),
            })
        self._human_capsules = capsules
        self._last_human_time = human_data_time(
            msg_world.header.stamp,
            self.get_clock().now(),
        )

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
        human_capsules = self._human_capsules_for_viz(
            self._current_human_capsules()
        )
        self.viz.publish(stamp, self.q_ctrl, self.q_legs)
        self.viz.publish_distances(
            stamp, self.q_ctrl, human_capsules or None, self.q_legs,
            self._active_external_pairs, self._active_internal_pairs,
        )

    def _human_capsules_for_viz(self, human_capsules_world):
        capsules = []
        if not human_capsules_world:
            return capsules
        pelvis_pose = self._lookup_pelvis_pose()
        if pelvis_pose is None:
            return capsules
        for capsule in human_capsules_world:
            capsules.append({
                'a': self._world_to_pelvis(capsule['a'], pelvis_pose),
                'b': self._world_to_pelvis(capsule['b'], pelvis_pose),
                'radius': capsule['radius'],
            })
        return capsules

    def _world_to_pelvis(self, point_world, pelvis_pose):
        return self._quat_rotate_np(
            self._quat_conjugate_np(pelvis_pose.quat),
            point_world - pelvis_pose.position,
        )

    def _lookup_pelvis_pose(self):
        pose, reason = self._tf_pose_lookup.lookup()
        if pose is None:
            self.get_logger().warn(
                '/human/colliders is in world frame, but TF '
                f'{self._tf_pose_lookup.describe()} is unavailable; '
                f'skipping human collider visualization for this tick: '
                f'{reason}',
                throttle_duration_sec=2.0,
            )
        return pose

    def _current_human_capsules(self):
        if not self._human_capsules:
            return []
        now = self.get_clock().now()
        if human_data_is_fresh(
            self._last_human_time,
            now,
            self._human_timeout_sec,
        ):
            return self._human_capsules
        return []

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
