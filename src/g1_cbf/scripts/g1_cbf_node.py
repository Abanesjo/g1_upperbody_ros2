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
import jax
import jax.numpy as jnp
import qpax
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

from cbfpy import CBF
from functools import partial

from g1_cbf.active_pairs import select_active_external_pairs_jax
from g1_cbf.cbf_config import G1CollisionCBFConfig
from g1_cbf.jax_kinematics import (
    CONTROLLED_JOINTS,
    LEG_JOINTS,
    N_BODIES,
    N_HUMAN_CAPSULES,
    N_LEG_JOINTS,
)
from g1_cbf_msg.msg import ActiveCollisionPairs, CapsuleArray


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
        self.declare_parameter('evaluate_at_actual', True)
        self.declare_parameter('collision_geometry', 'capsules')
        self.declare_parameter('sphere_interpolation_level', 0)
        self.declare_parameter('sphere_radius_gain', 1.0)
        self.declare_parameter('beta', 1.05)
        self.declare_parameter('publish_viz', False)
        self.declare_parameter('max_iter', 100)
        self.declare_parameter('solver_tol', 1e-3)
        self.declare_parameter('human_half_lengths', [0.33, 0.20, 0.20, 0.145, 0.145, 0.225, 0.225, 0.225, 0.225])
        self.declare_parameter('human_radii', [0.10, 0.05, 0.05, 0.05, 0.05, 0.065, 0.065, 0.065, 0.065])
        self.declare_parameter('human_radius_scale', 1.0)
        self.declare_parameter('external_filter_enabled', True)
        self.declare_parameter('external_activation_distance', 0.35)
        self.declare_parameter('external_max_active_pairs', 16)
        self.declare_parameter('external_always_keep_nearest', 4)

        dt = self.get_parameter('dt').value
        gamma = self.get_parameter('gamma').value
        margin_phi = self.get_parameter('margin_phi').value
        max_velocity = self.get_parameter('max_velocity').value
        geom = self.get_parameter('collision_geometry').value

        self.get_logger().info(
            f'CBF params: dt={dt}, gamma={gamma}, '
            f'margin_phi={margin_phi}, max_vel={max_velocity}, '
            f'geometry={geom}'
        )
        if self.get_parameter('publish_viz').value:
            self.get_logger().warn(
                'publish_viz on g1_cbf_node is ignored. Launch '
                'g1_cbf_viz_node for isolated, rate-limited visualization.'
            )

        # Build CBF (triggers JAX JIT warmup)
        self.get_logger().info('Initializing cbfpy CBF (JAX JIT warmup)...')
        human_radius_scale = float(self.get_parameter('human_radius_scale').value)
        human_radii = [
            float(r) * human_radius_scale
            for r in self.get_parameter('human_radii').value
        ]
        external_filter_enabled = bool(
            self.get_parameter('external_filter_enabled').value
        )
        external_max_active_pairs = int(
            self.get_parameter('external_max_active_pairs').value
        )
        full_external_pairs = N_BODIES * N_HUMAN_CAPSULES
        if external_filter_enabled:
            self._external_pair_slots = max(
                1, min(external_max_active_pairs, full_external_pairs)
            )
        else:
            self._external_pair_slots = full_external_pairs
        config = G1CollisionCBFConfig(
            gamma=gamma,
            margin_phi=margin_phi,
            max_velocity=max_velocity,
            collision_geometry=geom,
            sphere_interpolation_level=self.get_parameter('sphere_interpolation_level').value,
            sphere_radius_gain=self.get_parameter('sphere_radius_gain').value,
            beta=self.get_parameter('beta').value,
            solver_tol=self.get_parameter('solver_tol').value,
            human_half_lengths=list(self.get_parameter('human_half_lengths').value),
            human_radii=human_radii,
            external_pair_slots=self._external_pair_slots,
        )
        self.cbf = CBF.from_config(config)
        self._select_active_pairs_jit = jax.jit(partial(
            select_active_external_pairs_jax,
            external_pair_slots=self._external_pair_slots,
        ))

        # Patch safety_filter to pass max_iter (cbfpy doesn't expose it)
        max_iter = self.get_parameter('max_iter').value
        cbf_ref = self.cbf

        @jax.jit
        def _safety_filter(z, u_des, *args, **kwargs):
            P, q, A, b, G, h = cbf_ref.qp_data(z, u_des, *args, **kwargs)
            x_qp = qpax.solve_qp_elastic_primal(
                P, q, G, h,
                penalty=jnp.asarray(cbf_ref.constraint_relaxation_penalties),
                solver_tol=cbf_ref.solver_tol,
                max_iter=max_iter,
            )
            return x_qp[:cbf_ref.m]

        self.cbf.safety_filter = _safety_filter

        # Warmup call (JIT compilation)
        import time as _time
        _z = jnp.zeros(8)
        _u = jnp.zeros(8)
        _ql = jnp.zeros(N_LEG_JOINTS)
        _hc = jnp.zeros((N_HUMAN_CAPSULES, 7))
        _pairs = jnp.zeros((self._external_pair_slots, 2), dtype=jnp.int32)
        _pair_mask = jnp.zeros(self._external_pair_slots, dtype=bool)
        t0 = _time.monotonic()
        _ = self.cbf.safety_filter(_z, _u, _ql, _hc, _pairs, _pair_mask)
        jit_time = _time.monotonic() - t0
        self.get_logger().info(
            f'CBF ready — {config.num_cbf} constraints '
            f'({geom} mode), external_pair_slots={self._external_pair_slots}, '
            f'JIT compiled in {jit_time:.1f}s'
        )

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
        self.active_pairs_pub = self.create_publisher(
            ActiveCollisionPairs, '/cbf/active_external_pairs', sensor_qos,
        )

        # Timers
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

        # Pack for JAX — evaluate barriers at actual or target state
        if self.get_parameter('evaluate_at_actual').value:
            z_np = self.q_ctrl
        else:
            z_np = self.q_cbf_target
        z = jnp.array(z_np, dtype=jnp.float64)
        u_des = jnp.array(dq_ref, dtype=jnp.float64)
        q_legs_jnp = jnp.array(self.q_legs, dtype=jnp.float64)
        human_caps, human_mask = self._pack_human_capsules()
        pair_indices_jnp, pair_mask_jnp, pair_clearances_jnp = (
            self._select_active_external_pairs(
                z, q_legs_jnp, human_caps, human_mask,
            )
        )

        # Single CBF call — FK + proximity + QP all on GPU
        dq_safe_jnp = self.cbf.safety_filter(
            z, u_des, q_legs_jnp, human_caps, pair_indices_jnp, pair_mask_jnp,
        )
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
        self._publish_active_pairs(
            safe_msg.header.stamp,
            pair_indices_jnp,
            pair_mask_jnp,
            pair_clearances_jnp,
        )

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pack_human_capsules(self):
        """Pack human capsules into fixed-size jnp arrays."""
        buf = np.zeros((N_HUMAN_CAPSULES, 7))
        mask = np.zeros(N_HUMAN_CAPSULES, dtype=bool)
        count = min(len(self._human_capsules), N_HUMAN_CAPSULES)
        for i in range(count):
            c = self._human_capsules[i]
            buf[i, :3] = c['a']
            buf[i, 3:6] = c['b']
            buf[i, 6] = c['radius']
            mask[i] = True
        return (
            jnp.array(buf, dtype=jnp.float64),
            jnp.array(mask, dtype=bool),
        )

    def _select_active_external_pairs(self, z, q_legs, human_caps, human_mask):
        return self._select_active_pairs_jit(
            z,
            q_legs,
            human_caps,
            human_mask,
            jnp.array(
                self.get_parameter('external_activation_distance').value,
                dtype=jnp.float64,
            ),
            jnp.array(
                self.get_parameter('external_always_keep_nearest').value,
                dtype=jnp.int32,
            ),
            jnp.array(
                self.get_parameter('external_filter_enabled').value,
                dtype=bool,
            ),
        )

    def _publish_active_pairs(self, stamp, pair_indices, pair_mask, clearances):
        if self.active_pairs_pub.get_subscription_count() == 0:
            return
        pair_indices = np.asarray(pair_indices)
        pair_mask = np.asarray(pair_mask)
        clearances = np.asarray(clearances)
        active_count = int(np.count_nonzero(pair_mask))

        msg = ActiveCollisionPairs()
        msg.header.stamp = stamp
        msg.header.frame_id = 'pelvis'
        for slot in range(active_count):
            msg.robot_body_index.append(int(pair_indices[slot, 0]))
            msg.human_capsule_index.append(int(pair_indices[slot, 1]))
            msg.clearance.append(float(clearances[slot]))
        self.active_pairs_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = G1CBFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
