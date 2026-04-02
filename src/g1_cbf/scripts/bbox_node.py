#!/usr/bin/env python3
"""Visualize 3D bounding boxes from object detection as RViz markers.

Subscribes to /bbox_3d (Detection3DArray), publishes purple translucent
CUBE markers on /obstacle_markers.
"""

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3
from std_msgs.msg import ColorRGBA


class BBoxNode(Node):
    def __init__(self):
        super().__init__('bbox_node')

        self.pub = self.create_publisher(
            MarkerArray, '/obstacle_markers', 10,
        )
        self.create_subscription(
            Detection3DArray, '/bbox_3d',
            self._bbox_cb, 10,
        )

        self.get_logger().info('bbox_node ready')

    def _bbox_cb(self, msg: Detection3DArray):
        markers = MarkerArray()

        for i, det in enumerate(msg.detections):
            m = Marker()
            m.header = det.header if det.header.frame_id else msg.header
            m.ns = 'obstacles'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose = det.bbox.center
            m.scale = Vector3(
                x=det.bbox.size.x,
                y=det.bbox.size.y,
                z=det.bbox.size.z,
            )
            m.color = ColorRGBA(r=0.6, g=0.2, b=0.8, a=0.4)
            m.lifetime.sec = 0
            m.lifetime.nanosec = 200_000_000  # 200ms
            markers.markers.append(m)

        # Delete old markers if fewer detections this frame
        for j in range(len(msg.detections), 20):
            m = Marker()
            m.header = msg.header
            m.ns = 'obstacles'
            m.id = j
            m.action = Marker.DELETE
            markers.markers.append(m)

        self.pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = BBoxNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
