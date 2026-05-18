#!/usr/bin/env python3
"""Rate-limited CBF collision visualizer.

Runs outside the CBF control node so marker construction and RViz publishing
cannot block the safety-filter timer.
"""

import os
os.environ.setdefault('JAX_ENABLE_X64', '1')
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

from g1_cbf.collider_viz import ColliderVisualizer
from g1_cbf.jax_kinematics import CONTROLLED_JOINTS, LEG_JOINTS
from g1_cbf_msg.msg import ActiveCollisionPairs, CapsuleArray


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

        self.q_ctrl = None
        self.q_legs = np.zeros(len(LEG_JOINTS))
        self._human_capsules = []
        self._active_external_pairs = None
        self._active_internal_pairs = None

        self.viz = ColliderVisualizer(
            self,
            geometry_type=self.get_parameter('collision_geometry').value,
            sphere_interpolation_level=(
                self.get_parameter('sphere_interpolation_level').value
            ),
            sphere_radius_gain=self.get_parameter('sphere_radius_gain').value,
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
        capsules = []
        for c in msg.capsules:
            capsules.append({
                'a': np.array([c.a.x, c.a.y, c.a.z]),
                'b': np.array([c.b.x, c.b.y, c.b.z]),
                'radius': float(c.radius),
            })
        self._human_capsules = capsules

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

        stamp = self.get_clock().now().to_msg()
        self.viz.publish(stamp, self.q_ctrl, self.q_legs)
        self.viz.publish_distances(
            stamp, self.q_ctrl, self._human_capsules or None, self.q_legs,
            self._active_external_pairs, self._active_internal_pairs,
        )


def main(args=None):
    rclpy.init(args=args)
    node = G1CBFVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
