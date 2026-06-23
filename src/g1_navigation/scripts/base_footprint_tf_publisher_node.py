#!/usr/bin/env python3
"""Publish simulated base_footprint TF and odometry from /pose/pelvis."""

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

ODOM_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

PELVIS_HEIGHT = 0.784202


class BaseFootprintTfPublisherNode(Node):
    def __init__(self):
        super().__init__('base_footprint_tf_publisher_node')

        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/odom', ODOM_QOS)
        self.create_subscription(
            PoseStamped,
            '/pose/pelvis',
            self._pose_cb,
            SENSOR_QOS,
        )

        self.get_logger().info(
            'Publishing TF world -> base_footprint and /odom from /pose/pelvis'
        )

    def _pose_cb(self, msg: PoseStamped):
        q = msg.pose.orientation
        offset_x, offset_y, offset_z = self._rotate_pelvis_offset(q)
        base_x = msg.pose.position.x - offset_x
        base_y = msg.pose.position.y - offset_y
        base_z = msg.pose.position.z - offset_z

        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = 'world'
        transform.child_frame_id = 'base_footprint'
        transform.transform.translation.x = base_x
        transform.transform.translation.y = base_y
        transform.transform.translation.z = base_z
        transform.transform.rotation = q

        self.tf_broadcaster.sendTransform(transform)

        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = 'world'
        odom.child_frame_id = 'base_footprint'
        odom.pose.pose.position.x = base_x
        odom.pose.pose.position.y = base_y
        odom.pose.pose.position.z = base_z
        odom.pose.pose.orientation = q
        self.odom_pub.publish(odom)

    def _rotate_pelvis_offset(self, q):
        return (
            2.0 * PELVIS_HEIGHT * (q.x * q.z + q.w * q.y),
            2.0 * PELVIS_HEIGHT * (q.y * q.z - q.w * q.x),
            PELVIS_HEIGHT * (1.0 - 2.0 * (q.x * q.x + q.y * q.y)),
        )


def main(args=None):
    rclpy.init(args=args)
    node = BaseFootprintTfPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
