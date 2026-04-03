#!/usr/bin/env python3
"""Ghost robot publisher for visualizing unsafe commands.

Subscribes to /joint_commands_unsafe (may be partial, e.g. 8 upper body joints)
and /joint_states (full 29 DOF feedback). Merges them so the ghost always has
all 29 joints: unsafe values override the corresponding joints from feedback.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster


class GhostPublisherNode(Node):
    def __init__(self):
        super().__init__('ghost_publisher_node')

        self.declare_parameter('offset_y', -1.0)
        offset_y = self.get_parameter('offset_y').value

        self._latest_joint_states = {}  # name -> position (from feedback)

        self.js_pub = self.create_publisher(
            JointState, '/ghost/joint_states', SENSOR_QOS,
        )

        self.create_subscription(
            JointState, '/joint_commands_unsafe',
            self._unsafe_cb, SENSOR_QOS,
        )

        self.create_subscription(
            JointState, '/joint_states',
            self._joint_states_cb, SENSOR_QOS,
        )

        # Static TF: pelvis -> ghost/pelvis
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'pelvis'
        t.child_frame_id = 'ghost/pelvis'
        t.transform.translation.x = 0.0
        t.transform.translation.y = float(offset_y)
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

        self.get_logger().info(
            f'Ghost publisher ready (offset_y={offset_y})'
        )

    def _joint_states_cb(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self._latest_joint_states[name] = msg.position[i]

    def _unsafe_cb(self, msg: JointState):
        # Start from current joint states, override with unsafe commands
        merged = dict(self._latest_joint_states)
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                merged[name] = msg.position[i]

        if not merged:
            return

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(merged.keys())
        out.position = list(merged.values())
        self.js_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GhostPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
