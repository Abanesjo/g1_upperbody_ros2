#!/usr/bin/env python3
"""CBF safety filter node for G1 humanoid self-collision avoidance.

Subscribes to /joint_commands_unsafe, applies CBF-QP filtering to prevent
self-collisions between torso and arms, publishes safe commands on
/joint_commands at a fixed rate (1/dt Hz).

Supports capsule and box collision geometry via collision_geometry param.
"""

import os
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from vision_msgs.msg import Detection3DArray
from scipy.spatial.transform import Rotation as Rot
import tf2_ros

from g1_cbf.kinematics import G1Kinematics, CONTROLLED_JOINTS, COLLISION_PAIRS
from g1_cbf.qp_solver import CBFQPSolver
from g1_cbf.collider_viz import ColliderVisualizer


class G1CBFNode(Node):
    def __init__(self):
        super().__init__('g1_cbf_node')

        # Parameters
        self.declare_parameter('dt', 0.02)
        self.declare_parameter('margin_phi', 0.001)
        self.declare_parameter('beta', 1.05)
        self.declare_parameter('gamma', 5.0)
        self.declare_parameter('K', 5.0)
        self.declare_parameter('max_velocity', 0.5)
        self.declare_parameter('lpf_gain', 0.1)
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('collision_geometry', 'capsules')
        self.declare_parameter('sphere_interpolation_level', 0)
        self.declare_parameter('sphere_radius_gain', 1.0)
        self.declare_parameter('use_gpu', False)

        # Set JAX config before importing cbf module
        use_gpu = self.get_parameter('use_gpu').value
        os.environ['JAX_PLATFORMS'] = 'cuda' if use_gpu else 'cpu'
        os.environ['JAX_ENABLE_X64'] = '1'
        from g1_cbf.cbf import DpaxCapsuleCBF, DpaxBoxCBF, SphereCBF  # noqa: E402

        dt = self.get_parameter('dt').value
        gamma = self.get_parameter('gamma').value
        urdf_path = self.get_parameter('urdf_path').value
        self.geom_type = self.get_parameter('collision_geometry').value

        if not urdf_path:
            self.get_logger().fatal('urdf_path parameter is required')
            raise RuntimeError('urdf_path not set')

        self.get_logger().info(f'Loading URDF: {urdf_path}')
        self.get_logger().info(
            f'CBF params: dt={dt}, gamma={gamma}, '
            f'geometry={self.geom_type}'
        )

        # Subsystems
        self.kin = G1Kinematics(urdf_path)
        self.get_logger().info('Initializing dpax CBF (JAX JIT warmup)...')

        if self.geom_type == 'boxes':
            beta = self.get_parameter('beta').value
            self.cbf = DpaxBoxCBF(gamma=gamma, beta=beta)
        elif self.geom_type == 'spheres':
            margin_phi = self.get_parameter('margin_phi').value
            self.cbf = SphereCBF(gamma=gamma, margin_phi=margin_phi)
        else:
            margin_phi = self.get_parameter('margin_phi').value
            self.cbf = DpaxCapsuleCBF(
                gamma=gamma, margin_phi=margin_phi,
            )

        self.get_logger().info('dpax CBF ready')
        self.qp = CBFQPSolver(
            n_joints=self.kin.n_q,
            n_cbf=len(COLLISION_PAIRS),
        )
        self.viz = ColliderVisualizer(
            self, self.kin, geometry_type=self.geom_type,
        )

        # State
        self.q_full = None
        self.q_des_latest = None
        self.q_des_filtered = None
        self.q_cbf_target = None
        self._obstacles = []  # list of {center, rot, half_extents}
        self._zero_J6 = np.zeros((6, self.kin.n_q))

        # Passthrough state for non-controlled joints
        self._passthrough_names = []
        self._passthrough_positions = []
        self._passthrough_ctrl_indices = {}  # ctrl_joint_name -> msg index

        # TF for obstacle frame conversion
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Velocity limits (rad/s) from URDF (hardware constraints)
        self.dq_max = np.array([
            30.0, 30.0,                               # waist roll/pitch
            37.0, 37.0, 37.0,                         # L shoulder pitch/roll, elbow
            37.0, 37.0, 37.0,                         # R shoulder pitch/roll, elbow
        ])
        self.dq_min = -self.dq_max

        # Subscribers
        self.create_subscription(
            JointState, '/joint_states',
            self._joint_states_cb, 10,
        )
        self.create_subscription(
            JointState, '/joint_commands_unsafe',
            self._unsafe_cmd_cb, 10,
        )
        self.create_subscription(
            Detection3DArray, '/bbox_3d',
            self._bbox_cb, 10,
        )

        # Publisher
        self.cmd_pub = self.create_publisher(
            JointState, '/joint_commands', 10,
        )

        # Timer
        self.create_timer(dt, self._tick)

        self.get_logger().info(
            f'g1_cbf_node ready — publishing at {1.0/dt:.0f} Hz'
        )

    def _joint_states_cb(self, msg: JointState):
        self.q_full = self.kin.joint_names_to_q_full(
            list(msg.name), list(msg.position),
        )

    def _unsafe_cmd_cb(self, msg: JointState):
        q_des = self._extract_controlled(msg)
        if q_des is not None:
            self.q_des_latest = q_des
            # Store full message for passthrough of non-controlled joints
            self._passthrough_names = list(msg.name)
            self._passthrough_positions = list(msg.position)
            ctrl_set = set(CONTROLLED_JOINTS)
            self._passthrough_ctrl_indices = {
                name: i for i, name in enumerate(msg.name)
                if name in ctrl_set
            }

    def _tick(self):
        if self.q_full is None or self.q_des_latest is None:
            return

        dt = self.get_parameter('dt').value
        K = self.get_parameter('K').value
        max_vel = self.get_parameter('max_velocity').value
        lpf = self.get_parameter('lpf_gain').value
        q_ctrl = self.kin.extract_controlled(self.q_full)

        # Initialize targets on first tick
        if self.q_des_filtered is None:
            self.q_des_filtered = self.q_des_latest.copy()
        if self.q_cbf_target is None:
            self.q_cbf_target = q_ctrl.copy()
        if 0 < lpf < 1:
            self.q_des_filtered += lpf * (
                self.q_des_latest - self.q_des_filtered
            )
        else:
            self.q_des_filtered = self.q_des_latest.copy()

        # Proportional gain + velocity clamp
        dq_ref = K * (self.q_des_filtered - self.q_cbf_target)
        dq_ref = np.clip(dq_ref, -max_vel, max_vel)

        # Update FK
        self.kin.update(self.q_full)

        # Build CBF constraints
        constraints = []
        closest_points = []
        metric_min = float('inf')
        stamp = self.get_clock().now().to_msg()

        if self.geom_type == 'boxes':
            self.viz.publish(stamp)
            self._build_box_constraints(
                constraints, closest_points, metric_min,
            )
            self._build_obstacle_constraints(
                constraints, closest_points,
            )
        elif self.geom_type == 'spheres':
            sphere_data = self._build_sphere_constraints(
                constraints, closest_points, metric_min,
            )
            self.viz.publish_spheres(stamp, sphere_data)
        else:
            self.viz.publish(stamp)
            self._build_capsule_constraints(
                constraints, closest_points, metric_min,
            )

        # Publish distance lines
        self.viz.publish_distances(
            self.get_clock().now().to_msg(), closest_points,
        )

        # Solve QP
        dq_safe = self.qp.solve(
            dq_ref, constraints,
            self.dq_min, self.dq_max,
        )

        # Integrate safe velocity into persistent target
        self.q_cbf_target += dq_safe * dt

        # Clamp to prevent divergence
        max_lead = 0.5
        self.q_cbf_target = np.clip(
            self.q_cbf_target,
            q_ctrl - max_lead,
            q_ctrl + max_lead,
        )

        safe_msg = JointState()
        stamp = self.get_clock().now().to_msg()
        safe_msg.header.stamp = stamp

        if self._passthrough_names:
            # Publish all joints: passthrough non-controlled, CBF-filter controlled
            safe_msg.name = list(self._passthrough_names)
            safe_msg.position = list(self._passthrough_positions)
            safe_msg.velocity = [0.0] * len(self._passthrough_names)
            for i, jname in enumerate(CONTROLLED_JOINTS):
                if jname in self._passthrough_ctrl_indices:
                    idx = self._passthrough_ctrl_indices[jname]
                    safe_msg.position[idx] = float(self.q_cbf_target[i])
                    safe_msg.velocity[idx] = float(dq_safe[i])
        else:
            # Fallback: only controlled joints (legacy behaviour)
            safe_msg.name = list(CONTROLLED_JOINTS)
            safe_msg.position = self.q_cbf_target.tolist()
            safe_msg.velocity = dq_safe.tolist()

        self.cmd_pub.publish(safe_msg)

    def _build_capsule_constraints(self, constraints, closest_points,
                                   metric_min):
        endpoints = {}
        for name in self.kin.collision_bodies:
            a, b, J_a, J_b = self.kin.get_endpoint_jacobians(name)
            body = self.kin.collision_bodies[name]
            endpoints[name] = {
                'a': a, 'b': b,
                'J_a': J_a, 'J_b': J_b,
                'radius': body['radius'],
            }

        for pair in COLLISION_PAIRS:
            nameA, nameB = pair
            eA, eB = endpoints[nameA], endpoints[nameB]

            phi, A_row, b_val, p1, p2 = self.cbf.build_constraint(
                eA['radius'], eA['a'], eA['b'],
                eA['J_a'], eA['J_b'],
                eB['radius'], eB['a'], eB['b'],
                eB['J_a'], eB['J_b'],
            )
            constraints.append((A_row, b_val))
            closest_points.append((p1, p2))

            # margin_phi = self.get_parameter('margin_phi').value
            # if phi < 3.0 * margin_phi:
            #     self.get_logger().info(
            #         f'phi_min={phi:.6f} margin={margin_phi}',
            #         throttle_duration_sec=0.2,
            #     )

    def _build_sphere_constraints(self, constraints, closest_points,
                                  metric_min):
        interp_level = self.get_parameter('sphere_interpolation_level').value
        radius_gain = self.get_parameter('sphere_radius_gain').value

        sphere_data = {}
        for name in self.kin.collision_bodies:
            sphere_data[name] = self.kin.get_sphere_decomposition(
                name,
                interpolation_level=interp_level,
                radius_gain=radius_gain,
            )

        for nameA, nameB in COLLISION_PAIRS:
            sA, sB = sphere_data[nameA], sphere_data[nameB]
            pair_constraints, pair_points = self.cbf.build_constraints_batch(
                sA['radius'], sA['centers'], sA['jacobians'],
                sB['radius'], sB['centers'], sB['jacobians'],
            )
            constraints.extend(pair_constraints)
            closest_points.extend(pair_points)

        return sphere_data

    def _build_box_constraints(self, constraints, closest_points,
                               metric_min):
        for pair in COLLISION_PAIRS:
            nameA, nameB = pair
            bodyA = self.kin.collision_bodies[nameA]
            bodyB = self.kin.collision_bodies[nameB]
            centerA, rotA = self.kin.get_collision_pose(nameA)
            centerB, rotB = self.kin.get_collision_pose(nameB)
            J6_A = self.kin.get_collision_jacobian(nameA)
            J6_B = self.kin.get_collision_jacobian(nameB)

            alpha, A_row, b_val, p1, p2 = self.cbf.build_constraint(
                bodyA, centerA, rotA, J6_A,
                bodyB, centerB, rotB, J6_B,
            )
            constraints.append((A_row, b_val))
            closest_points.append((p1, p2))

            # beta = self.get_parameter('beta').value
            # if alpha < 1.5 * beta:
            #     self.get_logger().info(
            #         f'alpha={alpha:.4f} beta={beta}',
            #         throttle_duration_sec=0.2,
            #     )

    def _bbox_cb(self, msg: Detection3DArray):
        obstacles = []
        for det in msg.detections:
            frame_id = det.header.frame_id or msg.header.frame_id
            try:
                tf = self.tf_buffer.lookup_transform(
                    'pelvis', frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05),
                )
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                continue

            # Transform obstacle center to pelvis frame
            t = tf.transform
            tf_pos = np.array([
                t.translation.x, t.translation.y, t.translation.z,
            ])
            tf_rot = Rot.from_quat([
                t.rotation.x, t.rotation.y,
                t.rotation.z, t.rotation.w,
            ]).as_matrix()

            det_pos = np.array([
                det.bbox.center.position.x,
                det.bbox.center.position.y,
                det.bbox.center.position.z,
            ])
            det_rot = Rot.from_quat([
                det.bbox.center.orientation.x,
                det.bbox.center.orientation.y,
                det.bbox.center.orientation.z,
                det.bbox.center.orientation.w,
            ]).as_matrix()

            center = tf_rot @ det_pos + tf_pos
            rot = tf_rot @ det_rot
            half_extents = np.array([
                det.bbox.size.x / 2.0,
                det.bbox.size.y / 2.0,
                det.bbox.size.z / 2.0,
            ])

            obstacles.append({
                'center': center,
                'rot': rot,
                'half_extents': half_extents,
            })
        self._obstacles = obstacles

    def _build_obstacle_constraints(self, constraints, closest_points):
        if not self._obstacles or self.geom_type != 'boxes':
            return
        from g1_cbf.cbf import _box_b_from_half_extents

        beta = self.get_parameter('beta').value
        for obs in self._obstacles:
            obs_b = _box_b_from_half_extents(obs['half_extents'])
            for body_name in self.kin.collision_bodies:
                bodyA = self.kin.collision_bodies[body_name]
                centerA, rotA = self.kin.get_collision_pose(body_name)
                J6_A = self.kin.get_collision_jacobian(body_name)

                alpha, A_row, b_val, p1, p2 = self.cbf.build_constraint(
                    bodyA, centerA, rotA, J6_A,
                    None, obs['center'], obs['rot'], self._zero_J6,
                    b_override_B=obs_b,
                )
                constraints.append((A_row, b_val))
                closest_points.append((p1, p2))

                # if alpha < 1.5 * beta:
                #     self.get_logger().info(
                #         f'obstacle alpha={alpha:.4f} body={body_name}',
                #         throttle_duration_sec=0.2,
                #     )

    def _extract_controlled(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        q = np.zeros(self.kin.n_q)
        for i, jname in enumerate(CONTROLLED_JOINTS):
            if jname not in name_to_pos:
                self.get_logger().warn(
                    f'Joint {jname} missing from '
                    f'/joint_commands_unsafe, dropping',
                    throttle_duration_sec=2.0,
                )
                return None
            q[i] = name_to_pos[jname]
        return q


def main(args=None):
    rclpy.init(args=args)
    node = G1CBFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
