#!/usr/bin/env python3
"""Merge the colliders of several simulated humans into /human/colliders.

g1_human_manual.launch.py runs one g1_human_node per human, each publishing
its own CapsuleArray. The CBF nodes read a single /human/colliders, on which
a second publisher would overwrite the first (best effort, depth 1), so this
node keeps the latest array of every input topic and publishes the fresh
ones, concatenated, at a fixed rate.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from g1_cbf_msg.msg import CapsuleArray
from g1_human.collider_merge import ColliderMerger

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class ColliderMergeNode(Node):
    def __init__(self):
        super().__init__('collider_merge_node')

        self.declare_parameter('input_topics', ['/human_0/colliders'])
        self.declare_parameter('output_topic', '/human/colliders')
        self.declare_parameter('rate', 50.0)
        # Below the CBF nodes' human_timeout_sec (0.5 s), so a human that
        # stops publishing is dropped before it ages the merged stamp.
        self.declare_parameter('stale_timeout_sec', 0.2)

        input_topics = [
            str(topic) for topic in self.get_parameter('input_topics').value
        ]
        rate = float(self.get_parameter('rate').value)
        if not input_topics:
            raise ValueError('input_topics must not be empty')
        if rate <= 0.0:
            raise ValueError('rate must be positive')

        self._merger = ColliderMerger(
            len(input_topics),
            float(self.get_parameter('stale_timeout_sec').value),
        )
        self._pub = self.create_publisher(
            CapsuleArray, str(self.get_parameter('output_topic').value),
            SENSOR_QOS,
        )
        self._input_topics = []
        for index, topic in enumerate(input_topics):
            subscription = self.create_subscription(
                CapsuleArray, topic,
                lambda msg, index=index: self._merger.update(
                    index, msg, self.get_clock().now(),
                ),
                SENSOR_QOS,
            )
            if subscription.topic_name == self._pub.topic_name:
                raise ValueError(
                    f'input topic {topic} is the output topic'
                )
            self._input_topics.append(subscription.topic_name)

        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'Merging {", ".join(self._input_topics)} into '
            f'{self._pub.topic_name} at {rate:.0f} Hz'
        )

    def _tick(self):
        merged, skipped = self._merger.merge(self.get_clock().now())
        if skipped:
            self.get_logger().warn(
                'Leaving out ' + ', '.join(
                    f'{self._input_topics[index]} ({reason})'
                    for index, reason in skipped
                ),
                throttle_duration_sec=2.0,
            )
        if merged is not None:
            self._pub.publish(merged)


def main(args=None):
    rclpy.init(args=args)
    node = ColliderMergeNode()
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
