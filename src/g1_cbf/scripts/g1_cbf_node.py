#!/usr/bin/env python3
"""CBF safety filter node for G1 humanoid collision avoidance.

Uses cbfpy (JAX-based CBF-QP) with hardcoded JAX forward kinematics.
The entire pipeline (FK → proximity → QP) runs on GPU via JAX.

Subscribes to /joint_commands_unsafe, applies CBF-QP filtering,
publishes safe commands on /joint_commands at a fixed rate.
"""

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

from g1_cbf.active_pairs import (
    N_EXTERNAL_ROBOT_BODIES,
    make_internal_sphere_pair_indices,
    select_active_external_pairs_jax,
    select_active_internal_sphere_pairs_jax,
)
from g1_cbf.cbf_config import G1CollisionCBFConfig
from g1_cbf.jax_kinematics import (
    CONTROLLED_JOINTS,
    CONTROLLED_JOINT_DEFAULTS,
    LEG_JOINTS,
    N_HUMAN_CAPSULES,
    N_CONTROLLED_JOINTS,
    N_LEG_JOINTS,
    compute_sphere_counts,
)
from g1_cbf.tf_pose import (
    TfPoseLookup,
    normalize_frame,
    resolve_lookup_timeout_sec,
)
from g1_cbf_msg.msg import ActiveCollisionPairs, CapsuleArray

WORLD_FRAME = 'world'
PELVIS_FRAME = 'pelvis'


class G1CBFNode(Node):
    def __init__(self):
        super().__init__('g1_cbf_node')

        # Parameters
        self.declare_parameter('dt', 0.02)
        self.declare_parameter('internal_margin_phi', 0.001)
        self.declare_parameter('external_margin_phi', 0.001)
        self.declare_parameter('internal_gamma', 5.0)
        self.declare_parameter('external_gamma', 5.0)
        self.declare_parameter('K', 5.0)
        self.declare_parameter('max_velocity', 2.0)
        self.declare_parameter('lpf_gain', 0.0)
        self.declare_parameter('max_lead', 2.0)
        self.declare_parameter('evaluate_at_actual', True)
        self.declare_parameter('collision_geometry', 'capsules')
        self.declare_parameter('sphere_interpolation_level', 0)
        self.declare_parameter('sphere_radius_gain', 1.0)
        self.declare_parameter('publish_viz', False)
        self.declare_parameter('max_iter', 100)
        self.declare_parameter('solver_tol', 1e-3)
        self.declare_parameter('human_half_lengths', [
            0.33, 0.20, 0.20, 0.145, 0.145, 0.225, 0.225,
            0.16, 0.16, 0.225, 0.225,
        ])
        self.declare_parameter('human_radii', [
            0.10, 0.05, 0.05, 0.05, 0.05, 0.065, 0.065,
            0.10, 0.10, 0.065, 0.065,
        ])
        self.declare_parameter('human_radius_scale', 1.0)
        self.declare_parameter('external_filter_enabled', True)
        self.declare_parameter('external_activation_distance', 0.35)
        self.declare_parameter('external_max_active_pairs', 16)
        self.declare_parameter('external_always_keep_nearest', 4)
        self.declare_parameter('external_torso_margin_phi', 0.1)
        self.declare_parameter('external_torso_gamma', 0.3)
        self.declare_parameter('internal_filter_enabled', True)
        self.declare_parameter('internal_activation_distance', 0.20)
        self.declare_parameter('internal_max_active_pairs', 48)
        self.declare_parameter('internal_always_keep_nearest', 12)
        self.declare_parameter('area_cbf', True)
        self.declare_parameter('head_circle_cbf_enabled', True)
        self.declare_parameter('head_collider_radius', 0.3)
        self.declare_parameter('world_circle_radius', 3.0)
        self.declare_parameter('world_frame', WORLD_FRAME)
        self.declare_parameter('pelvis_frame', PELVIS_FRAME)
        self.declare_parameter('tf_lookup_timeout_sec', 0.0)
        self.declare_parameter('tf_timeout_sec', 0.0)

        dt = self.get_parameter('dt').value
        internal_gamma = float(self.get_parameter('internal_gamma').value)
        external_gamma = float(self.get_parameter('external_gamma').value)
        internal_margin_phi = float(
            self.get_parameter('internal_margin_phi').value
        )
        external_margin_phi = float(
            self.get_parameter('external_margin_phi').value
        )
        external_torso_margin_phi = float(
            self.get_parameter('external_torso_margin_phi').value
        )
        external_torso_gamma = float(
            self.get_parameter('external_torso_gamma').value
        )
        max_velocity = self.get_parameter('max_velocity').value
        head_collider_radius = float(
            self.get_parameter('head_collider_radius').value
        )
        world_circle_radius = float(
            self.get_parameter('world_circle_radius').value
        )
        area_cbf = bool(self.get_parameter('area_cbf').value)
        self._world_frame = normalize_frame(
            self.get_parameter('world_frame').value,
            WORLD_FRAME,
        )
        self._pelvis_frame = normalize_frame(
            self.get_parameter('pelvis_frame').value,
            PELVIS_FRAME,
        )
        self._tf_pose_lookup = TfPoseLookup(
            self,
            self._world_frame,
            self._pelvis_frame,
            resolve_lookup_timeout_sec(self),
        )
        geom = str(self.get_parameter('collision_geometry').value).lower()
        if geom not in ('capsules', 'spheres'):
            raise ValueError(
                "collision_geometry must be 'capsules' or 'spheres'"
            )
        if head_collider_radius <= 0.0:
            raise ValueError('head_collider_radius must be positive')
        if world_circle_radius <= 0.0:
            raise ValueError('world_circle_radius must be positive')

        self.get_logger().info(
            f'CBF params: dt={dt}, '
            f'internal_gamma={internal_gamma}, '
            f'external_gamma={external_gamma}, '
            f'external_torso_gamma={external_torso_gamma}, '
            f'internal_margin_phi={internal_margin_phi}, '
            f'external_margin_phi={external_margin_phi}, '
            f'external_torso_margin_phi={external_torso_margin_phi}, '
            f'head_circle_radius={world_circle_radius}, '
            f'head_collider_radius={head_collider_radius}, '
            f'area_cbf={area_cbf}, '
            f'max_vel={max_velocity}, geometry={geom}'
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
        sphere_interpolation_level = int(
            self.get_parameter('sphere_interpolation_level').value
        )
        sphere_radius_gain = float(
            self.get_parameter('sphere_radius_gain').value
        )
        full_external_pairs = N_EXTERNAL_ROBOT_BODIES * N_HUMAN_CAPSULES
        if external_filter_enabled:
            self._external_pair_slots = max(
                1, min(external_max_active_pairs, full_external_pairs)
            )
        else:
            self._external_pair_slots = full_external_pairs

        self._internal_pair_slots = 1
        self._all_internal_sphere_pair_indices = jnp.zeros((1, 4), dtype=jnp.int32)
        self._sphere_counts = compute_sphere_counts(sphere_interpolation_level)
        if geom == 'spheres':
            self._all_internal_sphere_pair_indices = make_internal_sphere_pair_indices(
                self._sphere_counts,
            )
            full_internal_pairs = int(self._all_internal_sphere_pair_indices.shape[0])
            internal_filter_enabled = bool(
                self.get_parameter('internal_filter_enabled').value
            )
            internal_max_active_pairs = int(
                self.get_parameter('internal_max_active_pairs').value
            )
            if internal_filter_enabled:
                self._internal_pair_slots = max(
                    1, min(internal_max_active_pairs, full_internal_pairs)
                )
            else:
                self._internal_pair_slots = full_internal_pairs
        config = G1CollisionCBFConfig(
            internal_gamma=internal_gamma,
            external_gamma=external_gamma,
            internal_margin_phi=internal_margin_phi,
            external_margin_phi=external_margin_phi,
            max_velocity=max_velocity,
            collision_geometry=geom,
            sphere_interpolation_level=sphere_interpolation_level,
            sphere_radius_gain=sphere_radius_gain,
            solver_tol=self.get_parameter('solver_tol').value,
            human_half_lengths=list(self.get_parameter('human_half_lengths').value),
            human_radii=human_radii,
            external_torso_margin_phi=external_torso_margin_phi,
            external_torso_gamma=external_torso_gamma,
            external_pair_slots=self._external_pair_slots,
            internal_pair_slots=self._internal_pair_slots,
        )
        self.cbf = CBF.from_config(config)
        self._select_active_external_pairs_jit = jax.jit(partial(
            select_active_external_pairs_jax,
            external_pair_slots=self._external_pair_slots,
        ))
        self._select_active_internal_pairs_jit = None
        if geom == 'spheres':
            self._select_active_internal_pairs_jit = jax.jit(partial(
                select_active_internal_sphere_pairs_jax,
                all_pair_indices=self._all_internal_sphere_pair_indices,
                sphere_counts=tuple(self._sphere_counts),
                max_robot_spheres=max(self._sphere_counts),
                sphere_radius_gain=sphere_radius_gain,
                internal_pair_slots=self._internal_pair_slots,
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
        _z = jnp.zeros(N_CONTROLLED_JOINTS)
        _u = jnp.zeros(N_CONTROLLED_JOINTS)
        _ql = jnp.zeros(N_LEG_JOINTS)
        _hc = jnp.zeros((N_HUMAN_CAPSULES, 7))
        _pairs = jnp.zeros((self._external_pair_slots, 2), dtype=jnp.int32)
        _pair_mask = jnp.zeros(self._external_pair_slots, dtype=bool)
        _internal_pairs = jnp.zeros((self._internal_pair_slots, 4), dtype=jnp.int32)
        _internal_mask = jnp.zeros(self._internal_pair_slots, dtype=bool)
        _pelvis_pos = jnp.zeros(3)
        _pelvis_quat = jnp.array([0.0, 0.0, 0.0, 1.0])
        _circle_radius = jnp.array(world_circle_radius, dtype=jnp.float64)
        _head_radius = jnp.array(head_collider_radius, dtype=jnp.float64)
        _head_circle_enabled = jnp.array(False)
        t0 = _time.monotonic()
        _ = self.cbf.safety_filter(
            _z, _u, _ql, _hc, _pairs, _pair_mask,
            _internal_pairs, _internal_mask, _pelvis_pos, _pelvis_quat,
            _circle_radius, _head_radius, _head_circle_enabled,
        )
        jit_time = _time.monotonic() - t0
        self.get_logger().info(
            f'CBF ready — {config.num_cbf} constraints '
            f'({geom} mode), external_pair_slots={self._external_pair_slots}, '
            f'internal_pair_slots={self._internal_pair_slots}, '
            f'JIT compiled in {jit_time:.1f}s'
        )

        # State
        self.q_ctrl = None   # (N_CONTROLLED_JOINTS,) current controlled joint positions
        self.q_legs = np.zeros(N_LEG_JOINTS)  # current leg joint positions
        self.q_des_latest = None
        self.q_des_filtered = None
        self.q_cbf_target = None
        self._human_capsules = []
        self._human_capsules_frame = self._pelvis_frame

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
            ActiveCollisionPairs, '/cbf/active_collision_pairs', sensor_qos,
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
        q = CONTROLLED_JOINT_DEFAULTS.copy()
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
        if self.q_des_latest is not None:
            q = self.q_des_latest.copy()
        elif self.q_ctrl is not None:
            q = self.q_ctrl.copy()
        else:
            q = CONTROLLED_JOINT_DEFAULTS.copy()

        updated = False
        for i, jname in enumerate(CONTROLLED_JOINTS):
            if jname in name_to_pos:
                q[i] = name_to_pos[jname]
                updated = True
        if not updated:
            self.get_logger().warn(
                '/joint_commands_unsafe contained no CBF-controlled joints',
                throttle_duration_sec=2.0,
            )
            return
        self.q_des_latest = q

    def _human_cb(self, msg: CapsuleArray):
        capsules = []
        for c in msg.capsules:
            capsules.append({
                'a': np.array([c.a.x, c.a.y, c.a.z]),
                'b': np.array([c.b.x, c.b.y, c.b.z]),
                'radius': c.radius,
            })
        self._human_capsules = capsules
        self._human_capsules_frame = self._normalize_frame(msg.header.frame_id)

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
        pelvis_pose = (
            self._lookup_pelvis_pose()
            if self._needs_pelvis_pose()
            else None
        )
        human_caps, human_mask = self._pack_human_capsules(pelvis_pose)
        pair_indices_jnp, pair_mask_jnp, pair_clearances_jnp = (
            self._select_active_external_pairs(
                z, q_legs_jnp, human_caps, human_mask,
            )
        )
        internal_indices_jnp, internal_mask_jnp, internal_clearances_jnp = (
            self._select_active_internal_pairs(z, q_legs_jnp)
        )
        (
            pelvis_position_jnp,
            pelvis_quat_jnp,
            world_circle_radius_jnp,
            head_collider_radius_jnp,
            head_circle_enabled_jnp,
        ) = self._head_circle_args(pelvis_pose)

        # Single CBF call — FK + proximity + QP all on GPU
        dq_safe_jnp = self.cbf.safety_filter(
            z, u_des, q_legs_jnp, human_caps, pair_indices_jnp, pair_mask_jnp,
            internal_indices_jnp, internal_mask_jnp, pelvis_position_jnp,
            pelvis_quat_jnp, world_circle_radius_jnp,
            head_collider_radius_jnp, head_circle_enabled_jnp,
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
            internal_indices_jnp,
            internal_mask_jnp,
            internal_clearances_jnp,
        )

        safe_msg.name = list(CONTROLLED_JOINTS)
        safe_msg.position = self.q_cbf_target.tolist()
        safe_msg.velocity = dq_safe.tolist()

        self.cmd_pub.publish(safe_msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pack_human_capsules(self, pelvis_pose):
        """Pack human capsules into fixed-size jnp arrays."""
        buf = np.zeros((N_HUMAN_CAPSULES, 7))
        mask = np.zeros(N_HUMAN_CAPSULES, dtype=bool)
        count = min(len(self._human_capsules), N_HUMAN_CAPSULES)
        if self._human_capsules_frame == self._world_frame and pelvis_pose is None:
            if count:
                self.get_logger().warn(
                    '/human/colliders is in world frame, but TF '
                    f'{self._tf_pose_lookup.describe()} is unavailable; '
                    'external human CBF disabled for this tick',
                    throttle_duration_sec=2.0,
                )
            return (
                jnp.array(buf, dtype=jnp.float64),
                jnp.array(mask, dtype=bool),
            )
        for i in range(count):
            c = self._human_capsules[i]
            a, b = self._capsule_endpoints_in_pelvis(c, pelvis_pose)
            buf[i, :3] = a
            buf[i, 3:6] = b
            buf[i, 6] = c['radius']
            mask[i] = True
        return (
            jnp.array(buf, dtype=jnp.float64),
            jnp.array(mask, dtype=bool),
        )

    def _capsule_endpoints_in_pelvis(self, capsule, pelvis_pose):
        frame = self._human_capsules_frame
        a = capsule['a']
        b = capsule['b']
        if frame == self._world_frame:
            return (
                self._world_to_pelvis(a, pelvis_pose),
                self._world_to_pelvis(b, pelvis_pose),
            )
        if frame not in ('', self._pelvis_frame):
            self.get_logger().warn(
                f"Unsupported /human/colliders frame '{frame}'; "
                "treating capsules as pelvis-frame coordinates",
                throttle_duration_sec=2.0,
            )
        return a, b

    def _world_to_pelvis(self, point_world, pelvis_pose):
        return self._quat_rotate_np(
            self._quat_conjugate_np(pelvis_pose.quat),
            point_world - pelvis_pose.position,
        )

    def _lookup_pelvis_pose(self):
        pose, reason = self._tf_pose_lookup.lookup()
        if pose is None:
            self.get_logger().warn(
                f'TF lookup failed for {self._tf_pose_lookup.describe()}: '
                f'{reason}',
                throttle_duration_sec=2.0,
            )
        return pose

    def _needs_pelvis_pose(self):
        head_circle_enabled = (
            bool(self.get_parameter('area_cbf').value)
            and bool(self.get_parameter('head_circle_cbf_enabled').value)
        )
        human_world_frame = (
            self._human_capsules_frame == self._world_frame
            and bool(self._human_capsules)
        )
        return head_circle_enabled or human_world_frame

    def _normalize_frame(self, frame_id):
        return normalize_frame(frame_id, self._pelvis_frame)

    @staticmethod
    def _normalize_quat(q):
        q = np.asarray(q, dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q / norm

    @staticmethod
    def _quat_conjugate_np(q):
        q = G1CBFNode._normalize_quat(q)
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

    @staticmethod
    def _quat_rotate_np(q, v):
        q = G1CBFNode._normalize_quat(q)
        xyz = q[:3]
        w = q[3]
        t = 2.0 * np.cross(xyz, v)
        return v + w * t + np.cross(xyz, t)

    def _select_active_external_pairs(self, z, q_legs, human_caps, human_mask):
        return self._select_active_external_pairs_jit(
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

    def _select_active_internal_pairs(self, z, q_legs):
        if self._select_active_internal_pairs_jit is None:
            return (
                jnp.zeros((self._internal_pair_slots, 4), dtype=jnp.int32),
                jnp.zeros(self._internal_pair_slots, dtype=bool),
                jnp.full(self._internal_pair_slots, jnp.inf, dtype=jnp.float64),
            )
        return self._select_active_internal_pairs_jit(
            z,
            q_legs,
            jnp.array(
                self.get_parameter('internal_activation_distance').value,
                dtype=jnp.float64,
            ),
            jnp.array(
                self.get_parameter('internal_always_keep_nearest').value,
                dtype=jnp.int32,
            ),
            jnp.array(
                self.get_parameter('internal_filter_enabled').value,
                dtype=bool,
            ),
        )

    def _head_circle_args(self, pelvis_pose):
        configured_enabled = (
            bool(self.get_parameter('area_cbf').value)
            and bool(self.get_parameter('head_circle_cbf_enabled').value)
        )
        enabled = configured_enabled and pelvis_pose is not None
        if configured_enabled and pelvis_pose is None:
            self.get_logger().warn(
                'head_circle_cbf_enabled is true, but TF '
                f'{self._tf_pose_lookup.describe()} is unavailable; '
                'head-circle CBF is disabled for this tick',
                throttle_duration_sec=2.0,
            )
        pelvis_position = (
            pelvis_pose.position if pelvis_pose is not None
            else np.zeros(3, dtype=np.float64)
        )
        pelvis_quat = (
            pelvis_pose.quat if pelvis_pose is not None
            else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        )
        return (
            jnp.array(pelvis_position, dtype=jnp.float64),
            jnp.array(pelvis_quat, dtype=jnp.float64),
            jnp.array(
                self.get_parameter('world_circle_radius').value,
                dtype=jnp.float64,
            ),
            jnp.array(
                self.get_parameter('head_collider_radius').value,
                dtype=jnp.float64,
            ),
            jnp.array(enabled, dtype=bool),
        )

    def _publish_active_pairs(self, stamp, pair_indices, pair_mask, clearances,
                              internal_indices, internal_mask,
                              internal_clearances):
        if self.active_pairs_pub.get_subscription_count() == 0:
            return
        pair_indices = np.asarray(pair_indices)
        pair_mask = np.asarray(pair_mask)
        clearances = np.asarray(clearances)
        active_count = int(np.count_nonzero(pair_mask))
        internal_indices = np.asarray(internal_indices)
        internal_mask = np.asarray(internal_mask)
        internal_clearances = np.asarray(internal_clearances)
        active_internal_count = int(np.count_nonzero(internal_mask))

        msg = ActiveCollisionPairs()
        msg.header.stamp = stamp
        msg.header.frame_id = 'pelvis'
        for slot in range(active_count):
            msg.robot_body_index.append(int(pair_indices[slot, 0]))
            msg.human_capsule_index.append(int(pair_indices[slot, 1]))
            msg.clearance.append(float(clearances[slot]))
        for slot in range(active_internal_count):
            msg.internal_body_a_index.append(int(internal_indices[slot, 0]))
            msg.internal_sphere_a_index.append(int(internal_indices[slot, 1]))
            msg.internal_body_b_index.append(int(internal_indices[slot, 2]))
            msg.internal_sphere_b_index.append(int(internal_indices[slot, 3]))
            msg.internal_clearance.append(float(internal_clearances[slot]))
        self.active_pairs_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = G1CBFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
