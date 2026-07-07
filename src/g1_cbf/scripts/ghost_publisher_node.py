#!/usr/bin/env python3
"""Ghost robot publisher for visualizing unsafe commands.

Subscribes to /joint_commands_unsafe (may be partial upper-body joints)
and /joint_states (full 29 DOF feedback). Merges them so the ghost always has
all 29 joints: unsafe values override the corresponding joints from feedback.
"""

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from tf2_ros import StaticTransformBroadcaster

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class GhostPublisherNode(Node):
    def __init__(self):
        super().__init__('ghost_publisher_node')

        self.declare_parameter('parent_frame', 'world')
        self.declare_parameter('child_frame', 'ghost/pelvis')
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('z', 0.78)
        self.declare_parameter('publish_rate', 50.0)
        parent_frame = self.get_parameter('parent_frame').value
        child_frame = self.get_parameter('child_frame').value
        x = float(self.get_parameter('x').value)
        y = float(self.get_parameter('y').value)
        z = float(self.get_parameter('z').value)
        publish_rate = float(self.get_parameter('publish_rate').value)
        if publish_rate <= 0.0:
            self.get_logger().warn('publish_rate must be positive; using 50 Hz')
            publish_rate = 50.0

        self._latest_joint_states = {}  # name -> position (from feedback)
        self._latest_unsafe_commands = {}  # name -> position (unsafe target)
        self._joint_state_order = []
        self._unsafe_order = []

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
        self.create_timer(1.0 / publish_rate, self._publish_ghost_state)

        # Static TF anchoring the ghost robot in the world frame.
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

        self.get_logger().info(
            'Ghost publisher ready '
            f'({parent_frame} -> {child_frame}, xyz=[{x:.3f}, {y:.3f}, {z:.3f}], '
            f'publish_rate={publish_rate:.1f} Hz)'
        )

    def _joint_states_cb(self, msg: JointState):
        self._joint_state_order = []
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self._joint_state_order.append(name)
                self._latest_joint_states[name] = msg.position[i]

    def _unsafe_cb(self, msg: JointState):
        unsafe_commands = {}
        unsafe_order = []
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                unsafe_order.append(name)
                unsafe_commands[name] = msg.position[i]
        self._unsafe_order = unsafe_order
        self._latest_unsafe_commands = unsafe_commands

    def _publish_ghost_state(self):
        merged = dict(self._latest_joint_states)
        merged.update(self._latest_unsafe_commands)

        if not merged:
            return

        names = [
            name for name in self._joint_state_order
            if name in merged
        ]
        names.extend(
            name for name in self._unsafe_order
            if name in merged and name not in names
        )

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = names
        out.position = [merged[name] for name in names]
        self.js_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GhostPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
