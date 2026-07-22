#!/usr/bin/env python3
"""CBF safety filter for planar base velocity commands."""

import time as _time

import jax
import jax.numpy as jnp
import numpy as np
import qpax
import rclpy
from cbfpy import CBF
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from tf2_ros import TransformException

from g1_cbf.cbf_config import G1CmdVelCBFConfig
from g1_cbf.human_capsules import (
    human_data_is_fresh,
    human_data_time,
    transform_capsule_array,
)
from g1_cbf.jax_kinematics import (
    CONTROLLED_JOINTS,
    LEG_JOINTS,
    N_HUMAN_CAPSULES,
    head_sphere_center_np,
)
from g1_cbf.tf_pose import (
    TfPoseLookup,
    normalize_frame,
    resolve_lookup_timeout_sec,
)
from g1_cbf_msg.msg import CapsuleArray


WORLD_FRAME = 'world'
PELVIS_FRAME = 'pelvis'
N_HUMAN_ENDPOINT_SPHERES = 2 * N_HUMAN_CAPSULES


class CmdVelCBFNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_cbf_node')

        self.declare_parameter('dt', 0.02)
        self.declare_parameter('area_cbf', True)
        self.declare_parameter('cmd_vel_cbf_enabled', True)
        self.declare_parameter('external_gamma', 5.0)
        self.declare_parameter('external_margin_phi', 0.001)
        self.declare_parameter('head_collider_radius', 0.3)
        self.declare_parameter('world_circle_radius', 3.0)
        self.declare_parameter('cmd_vel_limits.lin_vel_x', [-1.0, 2.0])
        self.declare_parameter('cmd_vel_limits.lin_vel_y', [-1.0, 1.0])
        self.declare_parameter('state_timeout_sec', 0.2)
        self.declare_parameter('human_timeout_sec', 0.5)
        self.declare_parameter('world_frame', WORLD_FRAME)
        self.declare_parameter('pelvis_frame', PELVIS_FRAME)
        self.declare_parameter('tf_lookup_timeout_sec', 0.0)
        self.declare_parameter('tf_stale_timeout_sec', 0.0)
        self.declare_parameter('tf_timeout_sec', 0.0)
        self.declare_parameter('max_iter', 100)
        self.declare_parameter('solver_tol', 1e-3)

        self._dt = float(self.get_parameter('dt').value)
        self._area_cbf = bool(self.get_parameter('area_cbf').value)
        self._external_gamma = float(
            self.get_parameter('external_gamma').value
        )
        self._external_margin_phi = float(
            self.get_parameter('external_margin_phi').value
        )
        self._head_collider_radius = float(
            self.get_parameter('head_collider_radius').value
        )
        self._world_circle_radius = float(
            self.get_parameter('world_circle_radius').value
        )
        self._lin_vel_x_limits = self._read_limit(
            'cmd_vel_limits.lin_vel_x'
        )
        self._lin_vel_y_limits = self._read_limit(
            'cmd_vel_limits.lin_vel_y'
        )
        self._state_timeout_sec = float(
            self.get_parameter('state_timeout_sec').value
        )
        self._human_timeout_sec = float(
            self.get_parameter('human_timeout_sec').value
        )
        self._tf_stale_timeout_sec = float(
            self.get_parameter('tf_stale_timeout_sec').value
        )
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
        self._max_iter = int(self.get_parameter('max_iter').value)
        self._solver_tol = float(self.get_parameter('solver_tol').value)

        self._validate_params()

        self.get_logger().info(
            'cmd_vel CBF params: '
            f'dt={self._dt}, gamma={self._external_gamma}, '
            f'margin={self._external_margin_phi}, '
            f'world_circle_radius={self._world_circle_radius}, '
            f'head_collider_radius={self._head_collider_radius}, '
            f'area_cbf={self._area_cbf}, '
            f'lin_vel_x={self._lin_vel_x_limits}, '
            f'lin_vel_y={self._lin_vel_y_limits}'
        )

        self._build_cbf()

        self._latest_cmd = Twist()
        self._pelvis_position = None
        self._pelvis_quat = None
        self._q_ctrl = None
        self._q_legs = None
        self._last_joint_time = None
        self._human_endpoint_points_xy = None
        self._human_endpoint_radii = None
        self._human_endpoint_mask = None
        self._last_human_time = None
        self._cbf_enabled = True

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        cbf_enable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, qos)
        self.create_subscription(
            JointState, '/joint_states', self._joint_states_cb, qos,
        )
        self.create_subscription(
            CapsuleArray, '/human/colliders', self._human_colliders_cb, qos,
        )
        self.create_subscription(
            Bool, '/cbf/enabled', self._cbf_enabled_cb, cbf_enable_qos,
        )
        self._cmd_pub = self.create_publisher(
            Twist, '/cmd_vel_safe', qos,
        )
        self.create_timer(self._dt, self._tick)

        self.get_logger().info(
            f'cmd_vel_cbf_node ready - publishing /cmd_vel_safe at '
            f'{1.0 / self._dt:.0f} Hz'
        )

    def _read_limit(self, name):
        values = list(self.get_parameter(name).value)
        if len(values) != 2:
            raise ValueError(f'{name} must have [min, max]')
        return [float(values[0]), float(values[1])]

    def _validate_params(self):
        if self._dt <= 0.0:
            raise ValueError('dt must be positive')
        if self._external_gamma <= 0.0:
            raise ValueError('external_gamma must be positive')
        if self._external_margin_phi < 0.0:
            raise ValueError('external_margin_phi must be non-negative')
        if self._head_collider_radius <= 0.0:
            raise ValueError('head_collider_radius must be positive')
        if self._world_circle_radius <= 0.0:
            raise ValueError('world_circle_radius must be positive')
        if self._lin_vel_x_limits[0] > self._lin_vel_x_limits[1]:
            raise ValueError('cmd_vel_limits.lin_vel_x min must be <= max')
        if self._lin_vel_y_limits[0] > self._lin_vel_y_limits[1]:
            raise ValueError('cmd_vel_limits.lin_vel_y min must be <= max')
        if self._state_timeout_sec < 0.0:
            raise ValueError('state_timeout_sec must be non-negative')
        if self._human_timeout_sec < 0.0:
            raise ValueError('human_timeout_sec must be non-negative')
        if self._tf_stale_timeout_sec < 0.0:
            raise ValueError('tf_stale_timeout_sec must be non-negative')
        if self._max_iter <= 0:
            raise ValueError('max_iter must be positive')
        if self._solver_tol <= 0.0:
            raise ValueError('solver_tol must be positive')

    def _build_cbf(self):
        self.get_logger().info(
            'Initializing cmd_vel CBF (JAX JIT warmup)...'
        )
        config = G1CmdVelCBFConfig(
            external_gamma=self._external_gamma,
            external_margin_phi=self._external_margin_phi,
            lin_vel_x_limits=self._lin_vel_x_limits,
            lin_vel_y_limits=self._lin_vel_y_limits,
            solver_tol=self._solver_tol,
        )
        self.cbf = CBF.from_config(config)
        cbf_ref = self.cbf
        max_iter = self._max_iter

        @jax.jit
        def _safety_filter(z, u_des, *args, **kwargs):
            P, q, A, b, G, h = cbf_ref.qp_data(z, u_des, *args, **kwargs)
            del A, b
            x_qp = qpax.solve_qp_elastic_primal(
                P, q, G, h,
                penalty=jnp.asarray(cbf_ref.constraint_relaxation_penalties),
                solver_tol=cbf_ref.solver_tol,
                max_iter=max_iter,
            )
            return x_qp[:cbf_ref.m]

        self.cbf.safety_filter = _safety_filter

        t0 = _time.monotonic()
        _ = self.cbf.safety_filter(
            jnp.array([1.0, 0.0]),
            jnp.zeros(2),
            jnp.array([0.0, 0.0, 0.0, 1.0]),
            jnp.array(self._world_circle_radius, dtype=jnp.float64),
            jnp.array(self._head_collider_radius, dtype=jnp.float64),
            jnp.array(True),
            jnp.zeros((N_HUMAN_ENDPOINT_SPHERES, 2), dtype=jnp.float64),
            jnp.zeros(N_HUMAN_ENDPOINT_SPHERES, dtype=jnp.float64),
            jnp.zeros(N_HUMAN_ENDPOINT_SPHERES, dtype=bool),
        )
        self.get_logger().info(
            f'cmd_vel CBF ready - JIT compiled in '
            f'{_time.monotonic() - t0:.1f}s'
        )

    def _cmd_vel_cb(self, msg: Twist):
        self._latest_cmd = msg

    def _cbf_enabled_cb(self, msg: Bool):
        self._cbf_enabled = bool(msg.data)

    def _joint_states_cb(self, msg: JointState):
        name_to_pos = {
            name: msg.position[i]
            for i, name in enumerate(msg.name)
            if i < len(msg.position)
        }
        required = list(CONTROLLED_JOINTS) + list(LEG_JOINTS)
        missing = [name for name in required if name not in name_to_pos]
        if missing:
            self.get_logger().warn(
                'Required joints missing from /joint_states; cmd_vel CBF '
                f'state not updated. First missing joint: {missing[0]}',
                throttle_duration_sec=2.0,
            )
            return

        self._q_ctrl = np.array(
            [name_to_pos[name] for name in CONTROLLED_JOINTS],
            dtype=np.float64,
        )
        self._q_legs = np.array(
            [name_to_pos[name] for name in LEG_JOINTS],
            dtype=np.float64,
        )
        self._last_joint_time = self.get_clock().now()

    def _human_colliders_cb(self, msg: CapsuleArray):
        try:
            msg_world = transform_capsule_array(
                msg,
                self._world_frame,
                self._tf_pose_lookup.buffer,
            )
        except (TransformException, ValueError) as exc:
            self.get_logger().warn(
                f"Invalid /human/colliders transform from "
                f"'{msg.header.frame_id}' to '{self._world_frame}': {exc}; "
                'keeping the last valid capsules',
                throttle_duration_sec=2.0,
            )
            return

        if not msg_world.capsules:
            self._clear_human_endpoints()
            self._last_human_time = human_data_time(
                msg_world.header.stamp,
                self.get_clock().now(),
            )
            return

        points_xy = np.zeros(
            (N_HUMAN_ENDPOINT_SPHERES, 2),
            dtype=np.float64,
        )
        radii = np.zeros(N_HUMAN_ENDPOINT_SPHERES, dtype=np.float64)
        mask = np.zeros(N_HUMAN_ENDPOINT_SPHERES, dtype=bool)

        for i, capsule in enumerate(
            msg_world.capsules[:N_HUMAN_CAPSULES]
        ):
            endpoint_xy = np.array([
                [capsule.a.x, capsule.a.y],
                [capsule.b.x, capsule.b.y],
            ], dtype=np.float64)
            radius = float(capsule.radius)

            slot = 2 * i
            points_xy[slot:slot + 2] = endpoint_xy
            radii[slot:slot + 2] = radius
            mask[slot:slot + 2] = True

        self._human_endpoint_points_xy = points_xy
        self._human_endpoint_radii = radii
        self._human_endpoint_mask = mask
        self._last_human_time = human_data_time(
            msg_world.header.stamp,
            self.get_clock().now(),
        )

    def _tick(self):
        cmd = self._latest_cmd
        safe_msg = Twist()
        safe_msg.angular.x = cmd.angular.x
        safe_msg.angular.y = cmd.angular.y
        safe_msg.angular.z = cmd.angular.z

        area_cbf = bool(self.get_parameter('area_cbf').value)
        cmd_vel_cbf_enabled = bool(
            self.get_parameter('cmd_vel_cbf_enabled').value
        )
        if not self._cbf_enabled or not area_cbf or not cmd_vel_cbf_enabled:
            safe_xy = self._clip_planar_velocity(
                np.array([cmd.linear.x, cmd.linear.y], dtype=np.float64)
            )
            safe_msg.linear.x = float(safe_xy[0])
            safe_msg.linear.y = float(safe_xy[1])
            safe_msg.linear.z = cmd.linear.z
            self._cmd_pub.publish(safe_msg)
            return

        state_ready, reason = self._state_ready()
        if not state_ready:
            self.get_logger().warn(
                f'cmd_vel CBF missing state ({reason}); zeroing linear '
                'velocity on /cmd_vel_safe',
                throttle_duration_sec=2.0,
            )
            self._cmd_pub.publish(safe_msg)
            return

        safe_xy = self._filter_planar_velocity(
            np.array([cmd.linear.x, cmd.linear.y], dtype=np.float64)
        )
        safe_msg.linear.x = float(safe_xy[0])
        safe_msg.linear.y = float(safe_xy[1])
        safe_msg.linear.z = cmd.linear.z
        self._cmd_pub.publish(safe_msg)

    def _state_ready(self):
        pose, reason = self._tf_pose_lookup.lookup()
        if pose is None:
            return (
                False,
                f'TF {self._tf_pose_lookup.describe()} unavailable: {reason}',
            )
        self._pelvis_position = pose.position
        self._pelvis_quat = pose.quat

        if self._q_ctrl is None or self._q_legs is None:
            return False, '/joint_states has not been received'
        now = self.get_clock().now()
        if self._tf_stale_timeout_sec > 0.0:
            pose_age = self._tf_pose_lookup.age_sec(pose)
            if pose_age is not None and pose_age > self._tf_stale_timeout_sec:
                return (
                    False,
                    f'TF {self._tf_pose_lookup.describe()} stale by '
                    f'{pose_age:.3f}s',
                )

        if self._state_timeout_sec == 0.0:
            return True, ''

        joint_age = (
            now - self._last_joint_time
        ).nanoseconds * 1e-9
        if joint_age > self._state_timeout_sec:
            return False, f'/joint_states stale by {joint_age:.3f}s'
        return True, ''

    def _filter_planar_velocity(self, u_des_np):
        head_xy = self._head_xy_world()
        (
            human_endpoint_points_xy,
            human_endpoint_radii,
            human_endpoint_mask,
        ) = self._human_endpoint_args()
        try:
            safe_jnp = self.cbf.safety_filter(
                jnp.array(head_xy, dtype=jnp.float64),
                jnp.array(u_des_np, dtype=jnp.float64),
                jnp.array(self._pelvis_quat, dtype=jnp.float64),
                jnp.array(self._world_circle_radius, dtype=jnp.float64),
                jnp.array(self._head_collider_radius, dtype=jnp.float64),
                jnp.array(True),
                jnp.array(human_endpoint_points_xy, dtype=jnp.float64),
                jnp.array(human_endpoint_radii, dtype=jnp.float64),
                jnp.array(human_endpoint_mask, dtype=bool),
            )
            safe = np.asarray(safe_jnp, dtype=np.float64)
        except Exception as exc:
            self.get_logger().error(
                f'cmd_vel CBF solve failed; zeroing planar velocity: {exc}',
                throttle_duration_sec=2.0,
            )
            return np.zeros(2, dtype=np.float64)

        if safe.shape != (2,) or not np.all(np.isfinite(safe)):
            self.get_logger().error(
                'cmd_vel CBF returned invalid planar velocity; zeroing',
                throttle_duration_sec=2.0,
            )
            return np.zeros(2, dtype=np.float64)
        return safe

    def _head_xy_world(self):
        head_pelvis = head_sphere_center_np(self._q_ctrl, self._q_legs)
        head_world = (
            self._pelvis_position
            + self._quat_rotate_np(self._pelvis_quat, head_pelvis)
        )
        return head_world[:2]

    def _human_endpoint_args(self):
        disabled = (
            np.zeros((N_HUMAN_ENDPOINT_SPHERES, 2), dtype=np.float64),
            np.zeros(N_HUMAN_ENDPOINT_SPHERES, dtype=np.float64),
            np.zeros(N_HUMAN_ENDPOINT_SPHERES, dtype=bool),
        )
        if (
            self._human_endpoint_points_xy is None
            or self._human_endpoint_radii is None
            or self._human_endpoint_mask is None
        ):
            return disabled

        now = self.get_clock().now()
        if not human_data_is_fresh(
            self._last_human_time,
            now,
            self._human_timeout_sec,
        ):
            age = (
                now - self._last_human_time
            ).nanoseconds * 1e-9
            self.get_logger().warn(
                f'/human/colliders stale by {age:.3f}s; '
                'head-vs-human cmd_vel CBF disabled for this tick',
                throttle_duration_sec=2.0,
            )
            return disabled

        return (
            self._human_endpoint_points_xy,
            self._human_endpoint_radii,
            self._human_endpoint_mask,
        )

    def _clear_human_endpoints(self):
        self._human_endpoint_points_xy = None
        self._human_endpoint_radii = None
        self._human_endpoint_mask = None

    def _clip_planar_velocity(self, u_np):
        return np.array([
            np.clip(
                u_np[0],
                self._lin_vel_x_limits[0],
                self._lin_vel_x_limits[1],
            ),
            np.clip(
                u_np[1],
                self._lin_vel_y_limits[0],
                self._lin_vel_y_limits[1],
            ),
        ], dtype=np.float64)

    @staticmethod
    def _quat_rotate_np(quat_xyzw, vec):
        quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
        norm = np.linalg.norm(quat_xyzw)
        if norm < 1e-9:
            return np.asarray(vec, dtype=np.float64)
        quat_xyzw = quat_xyzw / norm
        q_vec = quat_xyzw[:3]
        q_w = quat_xyzw[3]
        t = 2.0 * np.cross(q_vec, vec)
        return vec + q_w * t + np.cross(q_vec, t)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelCBFNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
