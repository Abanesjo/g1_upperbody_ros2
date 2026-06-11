#!/usr/bin/env python3
"""Broadcast world -> pelvis TF from /pose/pelvis."""

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class PelvisTfPublisherNode(Node):
    def __init__(self):
        super().__init__('pelvis_tf_publisher_node')

        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(
            PoseStamped,
            '/pose/pelvis',
            self._pose_cb,
            SENSOR_QOS,
        )

        self.get_logger().info(
            'Publishing TF world -> pelvis from /pose/pelvis'
        )

    def _pose_cb(self, msg: PoseStamped):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'world'
        t.child_frame_id = 'pelvis'
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = PelvisTfPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
