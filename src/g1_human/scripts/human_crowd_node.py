#!/usr/bin/env python3
"""A crowd of simulated humans walking at random inside a square.

Every human is simulated in this one process (one JAX runtime), and all of
their capsules go out as a single CapsuleArray on /human/colliders in the
world frame, so no merger is needed. Each human walks between random
waypoints (g1_human.crowd.CrowdSimulator); hips, knees, arms and torso move
with a walking gait whose amplitude follows the walking speed.

RViz markers go to /human/collider_markers at marker_rate, drawn as a few
line and sphere lists (one pair per capsule radius) so a hundred humans stay
cheap to send and draw.
"""

import jax
import jax.numpy as jnp
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from g1_cbf.jax_kinematics import N_HUMAN_CAPSULES, capsule_endpoints_all
from g1_cbf_msg.msg import Capsule, CapsuleArray
from g1_human.crowd import CrowdSimulator
from g1_human.human_body import HUMAN_CAPSULE_NAMES, human_capsules

WORLD_FRAME = 'world'
MARKER_NAMESPACE = 'human_crowd'
COLOR = ColorRGBA(r=0.9, g=0.5, b=0.1, a=0.6)  # orange, as g1_human_node
AREA_COLOR = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.8)

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class HumanCrowdNode(Node):
    def __init__(self):
        super().__init__('human_crowd_node')

        self.declare_parameter('num_humans', 20)
        self.declare_parameter('seed', 0)
        self.declare_parameter('publish_viz', True)
        side_length = self._float_parameter('side_length', 10.0)
        center = (
            self._float_parameter('center_x', 0.0),
            self._float_parameter('center_y', 0.0),
        )
        min_speed = self._float_parameter('min_speed', 0.4)
        max_speed = self._float_parameter('max_speed', 0.5)
        max_pause_sec = self._float_parameter('max_pause_sec', 2.0)
        spawn_keepout_radius = self._float_parameter('spawn_keepout_radius', 1.0)
        rate = self._float_parameter('rate', 30.0)
        marker_rate = self._float_parameter('marker_rate', 10.0)
        self._radius_scale = self._float_parameter('human_radius_scale', 1.0)
        if rate <= 0.0 or marker_rate <= 0.0:
            raise ValueError('rate and marker_rate must be positive')

        num_humans = int(self.get_parameter('num_humans').value)
        seed = int(self.get_parameter('seed').value)
        self._crowd = CrowdSimulator(
            num_humans, side_length, center,
            min_speed=min_speed,
            max_speed=max_speed,
            max_pause_sec=max_pause_sec,
            spawn_keepout_radius=spawn_keepout_radius,
            rng=np.random.default_rng(None if seed < 0 else seed),
        )
        capsule_count = num_humans * len(HUMAN_CAPSULE_NAMES)
        if capsule_count > N_HUMAN_CAPSULES:
            self.get_logger().warn(
                f'{num_humans} humans are {capsule_count} capsules; the CBF '
                f'nodes use only the first {N_HUMAN_CAPSULES} '
                '(N_HUMAN_CAPSULES)'
            )
        self._capsule_names = [
            f'human_{i}/{name}'
            for i in range(num_humans) for name in HUMAN_CAPSULE_NAMES
        ]

        # Forward kinematics of all humans in one compiled call.
        self._fk = jax.jit(jax.vmap(capsule_endpoints_all))
        self._capsules_world()  # compile now, not on the first tick

        self._capsule_pub = self.create_publisher(
            CapsuleArray, '/human/colliders', SENSOR_QOS,
        )
        self._period = 1.0 / rate
        self._last_tick = None
        self._latest = None
        self.create_timer(self._period, self._tick)

        if self.get_parameter('publish_viz').value:
            self._marker_pub = self.create_publisher(
                MarkerArray, '/human/collider_markers', 10,
            )
            self._marker_lifetime = Duration(
                seconds=max(0.5, 3.0 / marker_rate),
            ).to_msg()
            self._area_points = [
                Point(x=center[0] + 0.5 * side_length * sx,
                      y=center[1] + 0.5 * side_length * sy, z=0.0)
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))
            ]
            self.create_timer(1.0 / marker_rate, self._publish_markers)

        self.get_logger().info(
            f'{num_humans} humans walking in a {side_length:g} m square '
            f'around ({center[0]:g}, {center[1]:g}); publishing '
            f'{self._capsule_pub.topic_name} at {rate:g} Hz'
        )

    def _float_parameter(self, name, default):
        # Dynamic typing, so that `side_length:=10` (an integer) also works.
        self.declare_parameter(
            name, default, ParameterDescriptor(dynamic_typing=True),
        )
        return float(self.get_parameter(name).value)

    def _capsules_world(self):
        """World-frame capsules of every human: a, b (N, 11, 3), radii (11,)."""
        q, q_legs, knee_angles, pelvis_height = self._crowd.walking_pose()
        a_body, b_body = self._fk(
            jnp.asarray(q, dtype=jnp.float64),
            jnp.asarray(q_legs, dtype=jnp.float64),
        )
        a, b, radii, _ = human_capsules(
            np.asarray(a_body), np.asarray(b_body), knee_angles,
            self._radius_scale,
        )
        c = np.cos(self._crowd.heading)
        s = np.sin(self._crowd.heading)
        zeros = np.zeros_like(c)
        rotation = np.stack([
            np.stack([c, -s, zeros], axis=-1),
            np.stack([s, c, zeros], axis=-1),
            np.stack([zeros, zeros, np.ones_like(c)], axis=-1),
        ], axis=-2)
        offset = np.column_stack([self._crowd.position, pelvis_height])
        a_world = np.einsum('nij,nkj->nki', rotation, a) + offset[:, None, :]
        b_world = np.einsum('nij,nkj->nki', rotation, b) + offset[:, None, :]
        return a_world, b_world, radii

    def _tick(self):
        now = self.get_clock().now()
        if self._last_tick is None:
            dt = self._period
        else:
            # Bounded, so a stall or a jump of the simulation clock does not
            # teleport the crowd.
            dt = min(max(0.0, (now - self._last_tick).nanoseconds * 1e-9), 0.2)
        self._last_tick = now
        self._crowd.step(dt)
        a_world, b_world, radii = self._capsules_world()
        self._latest = (a_world, b_world, radii)

        msg = CapsuleArray()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = WORLD_FRAME
        radii_list = [float(r) for r in radii] * self._crowd.num_humans
        msg.capsules = [
            Capsule(
                a=Point(x=a[0], y=a[1], z=a[2]),
                b=Point(x=b[0], y=b[1], z=b[2]),
                radius=radius,
                name=name,
            )
            for a, b, radius, name in zip(
                a_world.reshape(-1, 3).tolist(),
                b_world.reshape(-1, 3).tolist(),
                radii_list,
                self._capsule_names,
            )
        ]
        self._capsule_pub.publish(msg)

    def _publish_markers(self):
        if self._latest is None:
            return
        a_world, b_world, radii = self._latest
        stamp = self.get_clock().now().to_msg()
        msg = MarkerArray()
        marker_id = 0
        for radius in np.unique(radii):
            same_radius = np.isclose(radii, radius)
            a = a_world[:, same_radius].reshape(-1, 3).tolist()
            b = b_world[:, same_radius].reshape(-1, 3).tolist()
            ends = [Point(x=x, y=y, z=z) for x, y, z in a + b]
            # A line as wide as the capsule between sphere caps draws the
            # capsule from any viewing angle.
            segments = [
                point for pair in zip(ends[:len(a)], ends[len(a):])
                for point in pair
            ]
            for marker_type, points in (
                (Marker.LINE_LIST, segments),
                (Marker.SPHERE_LIST, ends),
            ):
                marker = self._marker(stamp, marker_id, marker_type)
                marker.scale.x = marker.scale.y = marker.scale.z = float(
                    2.0 * radius
                )
                marker.points = points
                msg.markers.append(marker)
                marker_id += 1

        area = self._marker(stamp, marker_id, Marker.LINE_STRIP)
        area.scale.x = 0.03
        area.color = AREA_COLOR
        area.points = self._area_points
        msg.markers.append(area)
        self._marker_pub.publish(msg)

    def _marker(self, stamp, marker_id, marker_type):
        marker = Marker()
        marker.header.frame_id = WORLD_FRAME
        marker.header.stamp = stamp
        marker.ns = MARKER_NAMESPACE
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color = COLOR
        marker.lifetime = self._marker_lifetime
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = HumanCrowdNode()
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
