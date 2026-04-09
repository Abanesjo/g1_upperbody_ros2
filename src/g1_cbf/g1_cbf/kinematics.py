"""Pinocchio-based FK and Jacobian computation for G1 upper body colliders.

Loads the full 29-DOF URDF, computes FK/Jacobians, and extracts only
the 8 controlled joint columns. Handles collision body offsets.
"""

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation


CONTROLLED_JOINTS = [
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_elbow_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_elbow_joint',
]

_COLLISION_BODIES = {
    'torso': {
        'frame': 'torso_link',
        'offset_xyz': np.array([0.0, 0.0, 0.16]),
        'offset_rot': Rotation.identity(),
        'half_length': 0.33,
        'radius': 0.1,
    },
    'left_arm': {
        'frame': 'left_elbow_link',
        'offset_xyz': np.array([0.15, 0.001, -0.005]),
        # Align Z along arm (pitch 90), then spin cross-section 45 deg
        'offset_rot': Rotation.from_euler('y', np.pi / 2) * Rotation.from_euler('z', np.pi / 4),
        'half_length': 0.20,
        'radius': 0.05,
    },
    'right_arm': {
        'frame': 'right_elbow_link',
        'offset_xyz': np.array([0.15, -0.001, -0.005]),
        'offset_rot': Rotation.from_euler('y', np.pi / 2) * Rotation.from_euler('z', np.pi / 4),
        'half_length': 0.20,
        'radius': 0.05,
    },
    'left_shoulder': {
        'frame': 'left_shoulder_roll_link',
        'offset_xyz': np.array([0.0, 0.0, -0.09]),
        'offset_rot': Rotation.identity(),
        'half_length': 0.145,
        'radius': 0.05,
    },
    'right_shoulder': {
        'frame': 'right_shoulder_roll_link',
        'offset_xyz': np.array([0.0, 0.0, -0.09]),
        'offset_rot': Rotation.identity(),
        'half_length': 0.145,
        'radius': 0.05,
    },
    'left_thigh': {
        'frame': 'left_hip_yaw_link',
        'offset_xyz': np.array([0.0, 0.0, 0.03]),
        'offset_rot': Rotation.identity(),
        'half_length': 0.15,
        'radius': 0.065,
    },
    'right_thigh': {
        'frame': 'right_hip_yaw_link',
        'offset_xyz': np.array([0.0, 0.0, 0.03]),
        'offset_rot': Rotation.identity(),
        'half_length': 0.15,
        'radius': 0.065,
    },
}

COLLISION_PAIRS = [
    ('left_arm', 'right_arm'),
    ('left_arm', 'torso'),
    ('right_arm', 'torso'),
    ('left_arm', 'left_thigh'),
    ('left_arm', 'right_thigh'),
    ('right_arm', 'left_thigh'),
    ('right_arm', 'right_thigh'),
    ('left_shoulder', 'right_arm'),
    ('right_shoulder', 'left_arm'),
]


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


class G1Kinematics:
    """Pinocchio wrapper for FK and Jacobians at collision body centers."""

    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Resolve frame IDs for collision bodies
        self.frame_ids = {}
        for name, body in _COLLISION_BODIES.items():
            fid = self.model.getFrameId(body['frame'])
            if fid >= self.model.nframes:
                raise ValueError(f"Frame '{body['frame']}' not found in URDF")
            self.frame_ids[name] = fid

        # Precompute collision body offset SE3 transforms
        self.offset_se3 = {}
        for name, body in _COLLISION_BODIES.items():
            R_off = body['offset_rot'].as_matrix()
            self.offset_se3[name] = pin.SE3(R_off, body['offset_xyz'])

        # Map controlled joint names to pinocchio q-vector indices
        self.controlled_q_indices = []
        for jname in CONTROLLED_JOINTS:
            jid = self.model.getJointId(jname)
            if jid >= self.model.njoints:
                raise ValueError(f"Joint '{jname}' not found in URDF")
            idx_q = self.model.joints[jid].idx_q
            self.controlled_q_indices.append(idx_q)
        self.controlled_q_indices = np.array(self.controlled_q_indices)

        # Map controlled joint names to pinocchio v-vector indices (for Jacobian columns)
        self.controlled_v_indices = []
        for jname in CONTROLLED_JOINTS:
            jid = self.model.getJointId(jname)
            idx_v = self.model.joints[jid].idx_v
            self.controlled_v_indices.append(idx_v)
        self.controlled_v_indices = np.array(self.controlled_v_indices)

        # Store neutral q as default
        self.q_full = pin.neutral(self.model)

    @property
    def n_q(self) -> int:
        return len(CONTROLLED_JOINTS)

    @property
    def collision_bodies(self):
        return _COLLISION_BODIES

    def build_full_q(self, q_controlled: np.ndarray, q_current_full: np.ndarray) -> np.ndarray:
        """Insert 8 controlled values into the full pinocchio q vector."""
        q = q_current_full.copy()
        for i, idx in enumerate(self.controlled_q_indices):
            q[idx] = q_controlled[i]
        return q

    def extract_controlled(self, q_full: np.ndarray) -> np.ndarray:
        """Extract 8 controlled joint values from full q vector."""
        return q_full[self.controlled_q_indices]

    def update(self, q_full: np.ndarray):
        """Run FK and prepare Jacobian computation."""
        self.q_full = q_full.copy()
        pin.forwardKinematics(self.model, self.data, self.q_full)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, self.q_full)

    def get_collision_pose(self, body_name: str):
        """Get world-frame center (3,) and rotation (3,3) of collision ellipsoid."""
        fid = self.frame_ids[body_name]
        oMf = self.data.oMf[fid]
        T_world = oMf * self.offset_se3[body_name]
        return T_world.translation.copy(), T_world.rotation.copy()

    def get_endpoint_jacobians(self, body_name: str):
        """Get capsule endpoints and their Jacobians (3 x n_controlled each).

        Returns (a, b, J_a, J_b) where a/b are the capsule endpoints
        in world frame and J_a/J_b map controlled joint velocities to
        endpoint velocities.
        """
        fid = self.frame_ids[body_name]
        body = _COLLISION_BODIES[body_name]
        seg_half = body['half_length'] - body['radius']

        J_frame = pin.getFrameJacobian(
            self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED
        )

        oMf = self.data.oMf[fid]
        offset_world = oMf.rotation @ body['offset_xyz']

        # Offset-corrected translational Jacobian
        J_trans = J_frame[:3, :] - _skew(offset_world) @ J_frame[3:, :]
        J_rot = J_frame[3:, :]

        # Capsule direction in world frame (Z column of collision body rotation)
        T_world = oMf * self.offset_se3[body_name]
        v = T_world.rotation[:, 2]

        # Endpoint positions
        center = T_world.translation
        a = center + seg_half * v
        b = center - seg_half * v

        # Direction Jacobian: dv/dq = -skew(v) @ J_rot
        J_v = -_skew(v) @ J_rot

        # Endpoint Jacobians
        J_a = J_trans + seg_half * J_v
        J_b = J_trans - seg_half * J_v

        # Extract controlled columns
        return (
            a.copy(), b.copy(),
            J_a[:, self.controlled_v_indices],
            J_b[:, self.controlled_v_indices],
        )

    def get_sphere_decomposition(
        self,
        body_name: str,
        interpolation_level: int = 0,
        radius_gain: float = 1.0,
    ) -> dict:
        """Decompose a capsule into spheres along its axis.

        Parameters
        ----------
        body_name : str
            Name of the collision body to decompose.
        interpolation_level : int
            Number of extra spheres inserted between each adjacent base pair.
        radius_gain : float
            Scaling factor applied to all sphere radii.

        Returns
        -------
        dict with keys:
            centers : np.ndarray (n_spheres, 3)
            jacobians : np.ndarray (n_spheres, 3, n_controlled)
            radius : float
            n_spheres : int
        """
        body = _COLLISION_BODIES[body_name]
        h = body['half_length']
        R = body['radius']

        a, b, J_a, J_b = self.get_endpoint_jacobians(body_name)

        L = 2.0 * h
        n_base = max(1, round(L / (2.0 * R)))
        n_total = n_base + max(0, n_base - 1) * interpolation_level

        if n_total == 1:
            t_values = np.array([0.5])
        else:
            t_values = np.linspace(0.0, 1.0, n_total)

        # Vectorized interpolation: centers (n, 3), jacobians (n, 3, 8)
        t = t_values[:, None]  # (n, 1)
        centers = (1.0 - t) * b[None, :] + t * a[None, :]

        t_j = t_values[:, None, None]  # (n, 1, 1)
        jacobians = (1.0 - t_j) * J_b[None, :, :] + t_j * J_a[None, :, :]

        return {
            'centers': centers,
            'jacobians': jacobians,
            'radius': R * radius_gain,
            'n_spheres': n_total,
        }

    def get_collision_jacobian(self, body_name: str) -> np.ndarray:
        """Get 6 x 8 Jacobian at the collision body center.

        Returns Jacobian with rows [translational(3); rotational(3)]
        and columns for the 8 controlled joints only.
        Uses LOCAL_WORLD_ALIGNED frame with offset correction.
        """
        fid = self.frame_ids[body_name]
        body = _COLLISION_BODIES[body_name]

        # Full 6 x nv Jacobian at the link frame origin
        J_frame = pin.getFrameJacobian(
            self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED
        )

        # Offset correction: v_P = v_O + omega x (R @ offset)
        # => J_trans_P = J_trans_O - skew(R @ offset) @ J_rot
        oMf = self.data.oMf[fid]
        offset_world = oMf.rotation @ body['offset_xyz']
        S = _skew(offset_world)

        J_corrected = J_frame.copy()
        J_corrected[:3, :] -= S @ J_frame[3:, :]

        # Extract only controlled joint columns
        return J_corrected[:, self.controlled_v_indices]

    def joint_names_to_q_full(self, names: list, positions: list) -> np.ndarray:
        """Build full q vector from a JointState message's names/positions."""
        q = pin.neutral(self.model)
        for name, pos in zip(names, positions):
            try:
                jid = self.model.getJointId(name)
                if jid < self.model.njoints:
                    idx_q = self.model.joints[jid].idx_q
                    q[idx_q] = pos
            except Exception:
                pass
        return q
