#!/usr/bin/env python3
"""Draw ZED body-tracking skeletons as RViz markers.

Subscribes to the fused skeletons of zedx_driver_ros2
(zed_msgs/ObjectsStamped) and republishes them as a MarkerArray in the
skeleton's own frame; TF resolves that frame (fusion_world) for RViz.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from visualization_msgs.msg import MarkerArray
from zed_msgs.msg import ObjectsStamped

from g1_human.skeleton_markers import skeleton_markers

# The fusion node publishes skeletons with SensorDataQoS (best effort).
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class SkeletonVizNode(Node):
    def __init__(self):
        super().__init__('skeleton_viz_node')

        self.declare_parameter('skeleton_topic', '/zed_fusion/body_trk/skeletons')
        self.declare_parameter('marker_topic', '/human/skeleton_markers')
        self.declare_parameter('joint_radius', 0.04)
        self.declare_parameter('bone_width', 0.02)
        self.declare_parameter('lifetime_sec', 0.5)

        self._joint_radius = float(self.get_parameter('joint_radius').value)
        self._bone_width = float(self.get_parameter('bone_width').value)
        self._lifetime_sec = float(self.get_parameter('lifetime_sec').value)
        skeleton_topic = str(self.get_parameter('skeleton_topic').value)
        marker_topic = str(self.get_parameter('marker_topic').value)

        self._marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.create_subscription(
            ObjectsStamped, skeleton_topic, self._skeleton_callback, SENSOR_QOS,
        )
        self.get_logger().info(
            f'Drawing skeletons from {skeleton_topic} on {marker_topic}'
        )

    def _skeleton_callback(self, msg):
        self._marker_pub.publish(skeleton_markers(
            msg,
            joint_radius=self._joint_radius,
            bone_width=self._bone_width,
            lifetime_sec=self._lifetime_sec,
        ))


def main(args=None):
    rclpy.init(args=args)
    node = SkeletonVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
