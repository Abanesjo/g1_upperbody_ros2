#!/usr/bin/env python3
"""CBF safety filter for planar base velocity commands."""

import time as _time

import jax
import jax.numpy as jnp
import numpy as np
import qpax
import rclpy
from cbfpy import CBF
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState

from g1_cbf.cbf_config import G1CmdVelCBFConfig
from g1_cbf.jax_kinematics import (
    CONTROLLED_JOINTS,
    LEG_JOINTS,
    head_sphere_center_np,
)


class CmdVelCBFNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_cbf_node')

        self.declare_parameter('dt', 0.02)
        self.declare_parameter('cmd_vel_cbf_enabled', True)
        self.declare_parameter('external_gamma', 5.0)
        self.declare_parameter('external_margin_phi', 0.001)
        self.declare_parameter('head_collider_radius', 0.3)
        self.declare_parameter('world_circle_radius', 3.0)
        self.declare_parameter('cmd_vel_limits.lin_vel_x', [-0.5, 1.0])
        self.declare_parameter('cmd_vel_limits.lin_vel_y', [-0.5, 0.5])
        self.declare_parameter('state_timeout_sec', 0.2)
        self.declare_parameter('max_iter', 100)
        self.declare_parameter('solver_tol', 1e-3)

        self._dt = float(self.get_parameter('dt').value)
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
        self._max_iter = int(self.get_parameter('max_iter').value)
        self._solver_tol = float(self.get_parameter('solver_tol').value)

        self._validate_params()

        self.get_logger().info(
            'cmd_vel CBF params: '
            f'dt={self._dt}, gamma={self._external_gamma}, '
            f'margin={self._external_margin_phi}, '
            f'world_circle_radius={self._world_circle_radius}, '
            f'head_collider_radius={self._head_collider_radius}, '
            f'lin_vel_x={self._lin_vel_x_limits}, '
            f'lin_vel_y={self._lin_vel_y_limits}'
        )

        self._build_cbf()

        self._latest_cmd = Twist()
        self._pelvis_position = None
        self._pelvis_quat = None
        self._last_pose_time = None
        self._q_ctrl = None
        self._q_legs = None
        self._last_joint_time = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, qos)
        self.create_subscription(
            PoseStamped, '/pose/pelvis', self._pelvis_pose_cb, qos,
        )
        self.create_subscription(
            JointState, '/joint_states', self._joint_states_cb, qos,
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
        )
        self.get_logger().info(
            f'cmd_vel CBF ready - JIT compiled in '
            f'{_time.monotonic() - t0:.1f}s'
        )

    def _cmd_vel_cb(self, msg: Twist):
        self._latest_cmd = msg

    def _pelvis_pose_cb(self, msg: PoseStamped):
        self._pelvis_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=np.float64)
        self._pelvis_quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ], dtype=np.float64)
        self._last_pose_time = self.get_clock().now()

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

    def _tick(self):
        cmd = self._latest_cmd
        safe_msg = Twist()
        safe_msg.angular.x = cmd.angular.x
        safe_msg.angular.y = cmd.angular.y
        safe_msg.angular.z = cmd.angular.z

        if not bool(self.get_parameter('cmd_vel_cbf_enabled').value):
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
        if self._pelvis_position is None or self._pelvis_quat is None:
            return False, '/pose/pelvis has not been received'
        if self._q_ctrl is None or self._q_legs is None:
            return False, '/joint_states has not been received'
        if self._state_timeout_sec == 0.0:
            return True, ''

        now = self.get_clock().now()
        pose_age = (
            now - self._last_pose_time
        ).nanoseconds * 1e-9
        joint_age = (
            now - self._last_joint_time
        ).nanoseconds * 1e-9
        if pose_age > self._state_timeout_sec:
            return False, f'/pose/pelvis stale by {pose_age:.3f}s'
        if joint_age > self._state_timeout_sec:
            return False, f'/joint_states stale by {joint_age:.3f}s'
        return True, ''

    def _filter_planar_velocity(self, u_des_np):
        head_xy = self._head_xy_world()
        try:
            safe_jnp = self.cbf.safety_filter(
                jnp.array(head_xy, dtype=jnp.float64),
                jnp.array(u_des_np, dtype=jnp.float64),
                jnp.array(self._pelvis_quat, dtype=jnp.float64),
                jnp.array(self._world_circle_radius, dtype=jnp.float64),
                jnp.array(self._head_collider_radius, dtype=jnp.float64),
                jnp.array(True),
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
