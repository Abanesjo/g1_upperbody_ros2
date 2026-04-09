#!/usr/bin/env python3
"""CBF safety filter node for G1 humanoid collision avoidance.

Uses cbfpy (JAX-based CBF-QP) with hardcoded JAX forward kinematics.
The entire pipeline (FK → proximity → QP) runs on GPU via JAX.

Subscribes to /joint_commands_unsafe, applies CBF-QP filtering,
publishes safe commands on /joint_commands at a fixed rate.
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax.numpy as jnp
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

from cbfpy import CBF
from g1_cbf.cbf_config import G1CollisionCBFConfig
from g1_cbf.jax_kinematics import (
    CONTROLLED_JOINTS,
    LEG_JOINTS,
    N_HUMAN_CAPSULES,
    N_LEG_JOINTS,
    capsule_endpoints_np,
)
from g1_cbf.collider_viz import ColliderVisualizer
from g1_cbf_msg.msg import CapsuleArray


class G1CBFNode(Node):
    def __init__(self):
        super().__init__('g1_cbf_node')

        # Parameters
        self.declare_parameter('dt', 0.02)
        self.declare_parameter('margin_phi', 0.001)
        self.declare_parameter('gamma', 5.0)
        self.declare_parameter('K', 5.0)
        self.declare_parameter('max_velocity', 2.0)
        self.declare_parameter('lpf_gain', 0.0)
        self.declare_parameter('max_lead', 2.0)
        self.declare_parameter('publish_viz', False)

        dt = self.get_parameter('dt').value
        gamma = self.get_parameter('gamma').value
        margin_phi = self.get_parameter('margin_phi').value
        max_velocity = self.get_parameter('max_velocity').value

        self.get_logger().info(
            f'CBF params: dt={dt}, gamma={gamma}, '
            f'margin_phi={margin_phi}, max_vel={max_velocity}'
        )

        # Build CBF (triggers JAX JIT warmup)
        self.get_logger().info('Initializing cbfpy CBF (JAX JIT warmup)...')
        config = G1CollisionCBFConfig(
            gamma=gamma,
            margin_phi=margin_phi,
            max_velocity=max_velocity,
        )
        self.cbf = CBF.from_config(config)

        # Warmup call
        _z = jnp.zeros(8)
        _u = jnp.zeros(8)
        _ql = jnp.zeros(N_LEG_JOINTS)
        _hc = jnp.zeros((N_HUMAN_CAPSULES, 7))
        _hn = jnp.array(0)
        _ = self.cbf.safety_filter(_z, _u, _ql, _hc, _hn)
        self.get_logger().info('CBF ready')

        # State
        self.q_ctrl = None   # (8,) current controlled joint positions
        self.q_legs = np.zeros(N_LEG_JOINTS)  # (6,) current leg joint positions
        self.q_des_latest = None
        self.q_des_filtered = None
        self.q_cbf_target = None
        self._human_capsules = []

        # Passthrough state for non-controlled joints
        self._passthrough_names = []
        self._passthrough_positions = []
        self._passthrough_ctrl_indices = {}

        # Visualization
        self.viz = ColliderVisualizer(self)

        # QoS: best-effort, volatile, depth 1
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self.create_subscription(
            JointState, '/joint_states',
            self._joint_states_cb, sensor_qos,
        )
        self.create_subscription(
            JointState, '/joint_commands_unsafe',
            self._unsafe_cmd_cb, sensor_qos,
        )
        self.create_subscription(
            CapsuleArray, '/human/colliders',
            self._human_cb, sensor_qos,
        )

        # Publisher
        self.cmd_pub = self.create_publisher(
            JointState, '/joint_commands', sensor_qos,
        )

        # Timer
        self.create_timer(dt, self._tick)

        self.get_logger().info(
            f'g1_cbf_node ready — publishing at {1.0/dt:.0f} Hz'
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _joint_states_cb(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        q = np.zeros(8)
        for i, jname in enumerate(CONTROLLED_JOINTS):
            if jname in name_to_pos:
                q[i] = name_to_pos[jname]
        self.q_ctrl = q
        ql = np.zeros(N_LEG_JOINTS)
        for i, jname in enumerate(LEG_JOINTS):
            if jname in name_to_pos:
                ql[i] = name_to_pos[jname]
        self.q_legs = ql

    def _unsafe_cmd_cb(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        q = np.zeros(8)
        missing = False
        for i, jname in enumerate(CONTROLLED_JOINTS):
            if jname not in name_to_pos:
                self.get_logger().warn(
                    f'Joint {jname} missing from /joint_commands_unsafe',
                    throttle_duration_sec=2.0,
                )
                missing = True
                break
            q[i] = name_to_pos[jname]
        if missing:
            return
        self.q_des_latest = q

        # Store for passthrough
        ctrl_set = set(CONTROLLED_JOINTS)
        self._passthrough_names = list(msg.name)
        self._passthrough_positions = list(msg.position)
        self._passthrough_ctrl_indices = {
            name: i for i, name in enumerate(msg.name)
            if name in ctrl_set
        }

    def _human_cb(self, msg: CapsuleArray):
        capsules = []
        for c in msg.capsules:
            capsules.append({
                'a': np.array([c.a.x, c.a.y, c.a.z]),
                'b': np.array([c.b.x, c.b.y, c.b.z]),
                'radius': c.radius,
            })
        self._human_capsules = capsules

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def _tick(self):
        if self.q_ctrl is None or self.q_des_latest is None:
            return

        dt = self.get_parameter('dt').value
        K = self.get_parameter('K').value
        max_vel = self.get_parameter('max_velocity').value
        lpf = self.get_parameter('lpf_gain').value

        # Initialize on first tick
        if self.q_des_filtered is None:
            self.q_des_filtered = self.q_des_latest.copy()
        if self.q_cbf_target is None:
            self.q_cbf_target = self.q_ctrl.copy()

        # Low-pass filter on desired position
        if 0 < lpf < 1:
            self.q_des_filtered += lpf * (
                self.q_des_latest - self.q_des_filtered
            )
        else:
            self.q_des_filtered = self.q_des_latest.copy()

        # Reference velocity: track desired from persistent target
        dq_ref = K * (self.q_des_filtered - self.q_cbf_target)
        dq_ref = np.clip(dq_ref, -max_vel, max_vel)

        # Pack for JAX — evaluate barriers at ACTUAL state
        z = jnp.array(self.q_ctrl, dtype=jnp.float64)
        u_des = jnp.array(dq_ref, dtype=jnp.float64)
        q_legs_jnp = jnp.array(self.q_legs, dtype=jnp.float64)
        human_caps, human_count = self._pack_human_capsules()

        # Single CBF call — FK + proximity + QP all on GPU
        dq_safe_jnp = self.cbf.safety_filter(z, u_des, q_legs_jnp, human_caps, human_count)
        dq_safe = np.asarray(dq_safe_jnp)

        # Integrate safe velocity into persistent target
        self.q_cbf_target += dq_safe * dt

        # Clamp target to stay near actual state
        max_lead = self.get_parameter('max_lead').value
        self.q_cbf_target = np.clip(
            self.q_cbf_target,
            self.q_ctrl - max_lead,
            self.q_ctrl + max_lead,
        )

        # Publish safe command
        safe_msg = JointState()
        safe_msg.header.stamp = self.get_clock().now().to_msg()

        if self._passthrough_names:
            safe_msg.name = list(self._passthrough_names)
            safe_msg.position = list(self._passthrough_positions)
            safe_msg.velocity = [0.0] * len(self._passthrough_names)
            for i, jname in enumerate(CONTROLLED_JOINTS):
                if jname in self._passthrough_ctrl_indices:
                    idx = self._passthrough_ctrl_indices[jname]
                    safe_msg.position[idx] = float(self.q_cbf_target[i])
                    safe_msg.velocity[idx] = float(dq_safe[i])
        else:
            safe_msg.name = list(CONTROLLED_JOINTS)
            safe_msg.position = self.q_cbf_target.tolist()
            safe_msg.velocity = dq_safe.tolist()

        self.cmd_pub.publish(safe_msg)

        # Visualization at actual state (outside hot path)
        if self.get_parameter('publish_viz').value:
            stamp = self.get_clock().now().to_msg()
            self.viz.publish(stamp, self.q_ctrl, self.q_legs)
            self.viz.publish_distances(
                stamp, self.q_ctrl, self._human_capsules or None,
                self.q_legs,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pack_human_capsules(self):
        """Pack human capsules into fixed-size jnp arrays."""
        buf = np.zeros((N_HUMAN_CAPSULES, 7))
        count = min(len(self._human_capsules), N_HUMAN_CAPSULES)
        for i in range(count):
            c = self._human_capsules[i]
            buf[i, :3] = c['a']
            buf[i, 3:6] = c['b']
            buf[i, 6] = c['radius']
        return jnp.array(buf, dtype=jnp.float64), jnp.array(count)


def main(args=None):
    rclpy.init(args=args)
    node = G1CBFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
