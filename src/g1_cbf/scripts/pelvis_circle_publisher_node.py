#!/usr/bin/env python3
"""Publish the workspace boundary circle for RViz."""

import math

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


class PelvisCirclePublisherNode(Node):
    def __init__(self):
        super().__init__('pelvis_circle_publisher_node')

        self.declare_parameter('radius', 3.0)
        self.declare_parameter('world_circle_radius', -1.0)
        self.declare_parameter('height', 0.5)
        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('segments', 96)
        self.declare_parameter('line_width', 0.02)
        self.declare_parameter('workspace_frame', 'workspace')

        world_circle_radius = float(
            self.get_parameter('world_circle_radius').value
        )
        self.radius = (
            world_circle_radius
            if world_circle_radius > 0.0
            else float(self.get_parameter('radius').value)
        )
        self.height = float(self.get_parameter('height').value)
        self.segments = int(self.get_parameter('segments').value)
        self.line_width = float(self.get_parameter('line_width').value)
        self.workspace_frame = str(
            self.get_parameter('workspace_frame').value
        ).strip().lstrip('/')

        if self.radius <= 0.0:
            self.get_logger().warn('radius must be positive; using 3.0 m')
            self.radius = 3.0
        if self.segments < 3:
            self.get_logger().warn('segments must be at least 3; using 96')
            self.segments = 96
        if self.line_width <= 0.0:
            self.get_logger().warn('line_width must be positive; using 0.02 m')
            self.line_width = 0.02
        if not self.workspace_frame:
            self.get_logger().warn(
                'workspace_frame must not be empty; using workspace'
            )
            self.workspace_frame = 'workspace'

        publish_rate = float(self.get_parameter('publish_rate').value)
        if publish_rate <= 0.0:
            self.get_logger().warn('publish_rate must be positive; using 5 Hz')
            publish_rate = 5.0

        self.pub = self.create_publisher(Marker, '/pelvis_circle', 10)
        self.create_timer(1.0 / publish_rate, self._publish_circle)

        self.get_logger().info(
            f'Publishing pelvis circle: radius={self.radius:.2f} m, '
            f'height={self.height:.2f} m, '
            f'frame={self.workspace_frame}, '
            f'rate={publish_rate:.1f} Hz'
        )

    def _publish_circle(self):
        marker = Marker()
        marker.header.frame_id = self.workspace_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'pelvis_circle'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.line_width
        marker.color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.9)

        for i in range(self.segments + 1):
            theta = 2.0 * math.pi * i / self.segments
            marker.points.append(Point(
                x=self.radius * math.cos(theta),
                y=self.radius * math.sin(theta),
                z=self.height,
            ))

        self.pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = PelvisCirclePublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
