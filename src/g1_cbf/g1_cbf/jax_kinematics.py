"""Pure JAX forward kinematics for G1 upper body collision capsules.

Hardcodes the kinematic chain from the 29-DOF URDF, using only the 8
controlled joints. All functions are JIT-compatible (pure jnp, no side
effects). Uncontrolled joints (waist_yaw, shoulder_yaw, all leg joints)
are fixed at zero (neutral).

Joint index mapping for q (8,):
  0: waist_roll
  1: waist_pitch
  2: left_shoulder_pitch
  3: left_shoulder_roll
  4: left_elbow
  5: right_shoulder_pitch
  6: right_shoulder_roll
  7: right_elbow
"""

import numpy as np
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Rotation / transform helpers
# ---------------------------------------------------------------------------

def rot_x(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([
        [1.0, 0.0, 0.0],
        [0.0,  c,  -s ],
        [0.0,  s,   c ],
    ])


def rot_y(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([
        [ c,  0.0,  s ],
        [0.0, 1.0, 0.0],
        [-s,  0.0,  c ],
    ])


def rot_z(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([
        [ c,  -s,  0.0],
        [ s,   c,  0.0],
        [0.0, 0.0, 1.0],
    ])


def _rpy_to_rot(roll, pitch, yaw):
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def _make_T(R, t):
    """Build 4x4 homogeneous transform from 3x3 rotation + (3,) translation."""
    T = jnp.eye(4)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(jnp.asarray(t))
    return T


def _joint_T_x(origin_T, theta):
    """Apply a revolute joint about X: origin_T @ rot_x(theta)."""
    R = rot_x(theta)
    T = jnp.eye(4).at[:3, :3].set(R)
    return origin_T @ T


def _joint_T_y(origin_T, theta):
    R = rot_y(theta)
    T = jnp.eye(4).at[:3, :3].set(R)
    return origin_T @ T


def _joint_T_z(origin_T, theta):
    R = rot_z(theta)
    T = jnp.eye(4).at[:3, :3].set(R)
    return origin_T @ T


# ---------------------------------------------------------------------------
# Pre-computed constant transforms from URDF (computed once at import time)
# ---------------------------------------------------------------------------

_I3 = np.eye(3)
_pi = np.pi

def _np_rpy(r, p, y):
    """Numpy RPY → rotation at module load time."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler('xyz', [r, p, y]).as_matrix()

def _np_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

# --- Upper body chain ---

# pelvis → waist_yaw_joint (uncontrolled, q=0, axis=Z)
# origin: xyz="0 0 0" rpy="0 0 0"
_T_PELVIS_TO_WAIST_YAW = jnp.array(_np_T(_I3, [0.0, 0.0, 0.0]))

# waist_yaw_link → waist_roll_joint (q[0], axis=X)
# origin: xyz="-0.0039635 0 0.035" rpy="0 0 0"
_T_WAIST_YAW_TO_ROLL = jnp.array(_np_T(_I3, [-0.0039635, 0.0, 0.035]))

# waist_roll_link → waist_pitch_joint (q[1], axis=Y)
# origin: xyz="0 0 0.019" rpy="0 0 0"
_T_ROLL_TO_PITCH = jnp.array(_np_T(_I3, [0.0, 0.0, 0.019]))

# torso_link → left_shoulder_pitch_joint (q[2], axis=Y)
# origin: xyz="0.0039563 0.10022 0.23778" rpy="0.27931 5.49E-05 -0.00019159"
_T_TORSO_TO_L_PITCH = jnp.array(_np_T(
    _np_rpy(0.27931, 5.49e-05, -0.00019159),
    [0.0039563, 0.10022, 0.23778],
))

# left_shoulder_pitch_link → left_shoulder_roll_joint (q[3], axis=X)
# origin: xyz="0 0.038 -0.013831" rpy="-0.27925 0 0"
_T_L_PITCH_TO_L_ROLL = jnp.array(_np_T(
    _np_rpy(-0.27925, 0.0, 0.0),
    [0.0, 0.038, -0.013831],
))

# left_shoulder_roll_link → left_shoulder_yaw_joint (uncontrolled, q=0, axis=Z)
# origin: xyz="0 0.00624 -0.1032"
_T_L_ROLL_TO_L_YAW = jnp.array(_np_T(_I3, [0.0, 0.00624, -0.1032]))

# left_shoulder_yaw_link → left_elbow_joint (q[4], axis=Y)
# origin: xyz="0.015783 0 -0.080518"
_T_L_YAW_TO_L_ELBOW = jnp.array(_np_T(_I3, [0.015783, 0.0, -0.080518]))

# torso_link → right_shoulder_pitch_joint (q[5], axis=Y)
# origin: xyz="0.0039563 -0.10021 0.23778" rpy="-0.27931 5.49E-05 0.00019159"
_T_TORSO_TO_R_PITCH = jnp.array(_np_T(
    _np_rpy(-0.27931, 5.49e-05, 0.00019159),
    [0.0039563, -0.10021, 0.23778],
))

# right_shoulder_pitch_link → right_shoulder_roll_joint (q[6], axis=X)
# origin: xyz="0 -0.038 -0.013831" rpy="0.27925 0 0"
_T_R_PITCH_TO_R_ROLL = jnp.array(_np_T(
    _np_rpy(0.27925, 0.0, 0.0),
    [0.0, -0.038, -0.013831],
))

# right_shoulder_roll_link → right_shoulder_yaw_joint (uncontrolled, q=0, axis=Z)
# origin: xyz="0 -0.00624 -0.1032"
_T_R_ROLL_TO_R_YAW = jnp.array(_np_T(_I3, [0.0, -0.00624, -0.1032]))

# right_shoulder_yaw_link → right_elbow_joint (q[7], axis=Y)
# origin: xyz="0.015783 0 -0.080518"
_T_R_YAW_TO_R_ELBOW = jnp.array(_np_T(_I3, [0.015783, 0.0, -0.080518]))

# --- Leg chain joint origins ---
# q_legs[0..5] = L_hip_pitch, L_hip_roll, L_hip_yaw, R_hip_pitch, R_hip_roll, R_hip_yaw

# pelvis → left_hip_pitch_joint (q_legs[0], axis=Y)
_T_PELVIS_TO_L_HIP_PITCH = jnp.array(_np_T(_I3, [0.0, 0.064452, -0.1027]))
# left_hip_pitch_link → left_hip_roll_joint (q_legs[1], axis=X)
_T_L_HIP_PITCH_TO_ROLL = jnp.array(_np_T(
    _np_rpy(0.0, -0.1749, 0.0), [0.0, 0.052, -0.030465],
))
# left_hip_roll_link → left_hip_yaw_joint (q_legs[2], axis=Z)
_T_L_HIP_ROLL_TO_YAW = jnp.array(_np_T(_I3, [0.025001, 0.0, -0.12412]))

# pelvis → right_hip_pitch_joint (q_legs[3], axis=Y)
_T_PELVIS_TO_R_HIP_PITCH = jnp.array(_np_T(_I3, [0.0, -0.064452, -0.1027]))
# right_hip_pitch_link → right_hip_roll_joint (q_legs[4], axis=X)
_T_R_HIP_PITCH_TO_ROLL = jnp.array(_np_T(
    _np_rpy(0.0, -0.1749, 0.0), [0.0, -0.052, -0.030465],
))
# right_hip_roll_link → right_hip_yaw_joint (q_legs[5], axis=Z)
_T_R_HIP_ROLL_TO_YAW = jnp.array(_np_T(_I3, [0.025001, 0.0, -0.12412]))

# --- Collision body offsets ---

def _np_offset_T(R, t):
    return _np_T(R, t)

_OFFSET_TORSO = jnp.array(_np_offset_T(_I3, [0.0, 0.0, 0.16]))
_OFFSET_L_ARM = jnp.array(_np_offset_T(
    _np_rpy(0.0, _pi / 2, 0.0) @ _np_rpy(0.0, 0.0, _pi / 4),
    [0.15, 0.001, -0.005],
))
_OFFSET_R_ARM = jnp.array(_np_offset_T(
    _np_rpy(0.0, _pi / 2, 0.0) @ _np_rpy(0.0, 0.0, _pi / 4),
    [0.15, -0.001, -0.005],
))
_OFFSET_L_SHOULDER = jnp.array(_np_offset_T(_I3, [0.0, 0.0, -0.09]))
_OFFSET_R_SHOULDER = jnp.array(_np_offset_T(_I3, [0.0, 0.0, -0.09]))
_OFFSET_L_THIGH = jnp.array(_np_offset_T(_I3, [0.0, 0.0, 0.03]))
_OFFSET_R_THIGH = jnp.array(_np_offset_T(_I3, [0.0, 0.0, 0.03]))

# Body ordering (fixed, matches array indices)
BODY_NAMES = [
    'torso', 'left_arm', 'right_arm',
    'left_shoulder', 'right_shoulder',
    'left_thigh', 'right_thigh',
]
N_BODIES = len(BODY_NAMES)
BODY_INDEX = {name: i for i, name in enumerate(BODY_NAMES)}

# (half_length, radius) per body
HALF_LENGTHS = jnp.array([0.33, 0.20, 0.20, 0.145, 0.145, 0.15, 0.15])
RADII = jnp.array([0.1, 0.05, 0.05, 0.05, 0.05, 0.065, 0.065])

# Collision pairs as index tuples
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
COLLISION_PAIR_INDICES = [
    (BODY_INDEX[a], BODY_INDEX[b]) for a, b in COLLISION_PAIRS
]
N_SELF_PAIRS = len(COLLISION_PAIR_INDICES)

# Controlled joint names (for ROS2 message parsing)
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

# Leg joint names (for extracting from /joint_states)
LEG_JOINTS = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
]
N_LEG_JOINTS = len(LEG_JOINTS)

# Maximum human capsules (for fixed-size arrays)
N_HUMAN_CAPSULES = 9


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def fk_body_transforms(q, q_legs):
    """Compute 4x4 world-frame transforms for each collision body's offset frame.

    Args:
        q: (8,) controlled joint positions.
        q_legs: (6,) leg joint positions [L_hip_pitch, L_hip_roll, L_hip_yaw,
                R_hip_pitch, R_hip_roll, R_hip_yaw].

    Returns:
        Tuple of 7 transforms (4x4 each), ordered per BODY_NAMES.
    """
    # Waist chain: pelvis → waist_yaw(0) → waist_roll(q0) → waist_pitch(q1) → torso
    T = _T_PELVIS_TO_WAIST_YAW  # waist_yaw origin (identity)
    # waist_yaw is uncontrolled (fixed at 0), so rot_z(0) = I — skip
    T = T @ _T_WAIST_YAW_TO_ROLL
    T = _joint_T_x(T, q[0])     # waist_roll
    T = T @ _T_ROLL_TO_PITCH
    T_torso_frame = _joint_T_y(T, q[1])  # waist_pitch → torso_link
    T_torso = T_torso_frame @ _OFFSET_TORSO

    # Left arm chain: torso → L_shoulder_pitch(q2) → L_shoulder_roll(q3) → L_shoulder_yaw(0) → L_elbow(q4)
    T = T_torso_frame @ _T_TORSO_TO_L_PITCH
    T = _joint_T_y(T, q[2])     # left_shoulder_pitch
    T = T @ _T_L_PITCH_TO_L_ROLL
    T_l_shoulder_frame = _joint_T_x(T, q[3])  # left_shoulder_roll → left_shoulder_roll_link
    T_l_shoulder = T_l_shoulder_frame @ _OFFSET_L_SHOULDER

    T = T_l_shoulder_frame @ _T_L_ROLL_TO_L_YAW
    # left_shoulder_yaw uncontrolled (fixed at 0) — skip rot_z(0)
    T = T @ _T_L_YAW_TO_L_ELBOW
    T_l_elbow_frame = _joint_T_y(T, q[4])  # left_elbow → left_elbow_link
    T_l_arm = T_l_elbow_frame @ _OFFSET_L_ARM

    # Right arm chain: torso → R_shoulder_pitch(q5) → R_shoulder_roll(q6) → R_shoulder_yaw(0) → R_elbow(q7)
    T = T_torso_frame @ _T_TORSO_TO_R_PITCH
    T = _joint_T_y(T, q[5])     # right_shoulder_pitch
    T = T @ _T_R_PITCH_TO_R_ROLL
    T_r_shoulder_frame = _joint_T_x(T, q[6])  # right_shoulder_roll → right_shoulder_roll_link
    T_r_shoulder = T_r_shoulder_frame @ _OFFSET_R_SHOULDER

    T = T_r_shoulder_frame @ _T_R_ROLL_TO_R_YAW
    # right_shoulder_yaw uncontrolled — skip
    T = T @ _T_R_YAW_TO_R_ELBOW
    T_r_elbow_frame = _joint_T_y(T, q[7])  # right_elbow → right_elbow_link
    T_r_arm = T_r_elbow_frame @ _OFFSET_R_ARM

    # Left leg: pelvis → hip_pitch(q_legs[0]) → hip_roll(q_legs[1]) → hip_yaw(q_legs[2])
    T = _T_PELVIS_TO_L_HIP_PITCH
    T = _joint_T_y(T, q_legs[0])
    T = T @ _T_L_HIP_PITCH_TO_ROLL
    T = _joint_T_x(T, q_legs[1])
    T = T @ _T_L_HIP_ROLL_TO_YAW
    T_l_hip_yaw = _joint_T_z(T, q_legs[2])
    T_l_thigh = T_l_hip_yaw @ _OFFSET_L_THIGH

    # Right leg: pelvis → hip_pitch(q_legs[3]) → hip_roll(q_legs[4]) → hip_yaw(q_legs[5])
    T = _T_PELVIS_TO_R_HIP_PITCH
    T = _joint_T_y(T, q_legs[3])
    T = T @ _T_R_HIP_PITCH_TO_ROLL
    T = _joint_T_x(T, q_legs[4])
    T = T @ _T_R_HIP_ROLL_TO_YAW
    T_r_hip_yaw = _joint_T_z(T, q_legs[5])
    T_r_thigh = T_r_hip_yaw @ _OFFSET_R_THIGH

    return (T_torso, T_l_arm, T_r_arm, T_l_shoulder, T_r_shoulder, T_l_thigh, T_r_thigh)


def capsule_endpoints_all(q, q_legs=None):
    """Compute all capsule endpoints in the pelvis frame.

    Args:
        q: (8,) controlled joint positions.
        q_legs: (6,) leg joint positions, or None for zeros (neutral).

    Returns:
        a_all: (N_BODIES, 3) — endpoint a for each capsule.
        b_all: (N_BODIES, 3) — endpoint b for each capsule.
    """
    if q_legs is None:
        q_legs = jnp.zeros(N_LEG_JOINTS)
    transforms = fk_body_transforms(q, q_legs)

    a_list = []
    b_list = []
    for i, T in enumerate(transforms):
        center = T[:3, 3]
        z_axis = T[:3, 2]
        seg_half = HALF_LENGTHS[i] - RADII[i]
        a_list.append(center + seg_half * z_axis)
        b_list.append(center - seg_half * z_axis)

    return jnp.stack(a_list), jnp.stack(b_list)


def capsule_endpoints_np(q_np, q_legs_np=None):
    """Numpy convenience wrapper for visualization (outside JIT path).

    Args:
        q_np: (8,) numpy array of controlled joint positions.
        q_legs_np: (6,) numpy array of leg joint positions, or None for zeros.

    Returns:
        a_all: (N_BODIES, 3) numpy
        b_all: (N_BODIES, 3) numpy
        radii: (N_BODIES,) numpy
    """
    q_j = jnp.array(q_np, dtype=jnp.float64)
    ql_j = jnp.array(q_legs_np, dtype=jnp.float64) if q_legs_np is not None else None
    a, b = capsule_endpoints_all(q_j, ql_j)
    return np.asarray(a), np.asarray(b), np.asarray(RADII)


# ---------------------------------------------------------------------------
# Sphere decomposition helpers
# ---------------------------------------------------------------------------

def compute_sphere_counts(interpolation_level=0):
    """Return n_spheres per body as a Python list (constant for JIT tracing).

    Args:
        interpolation_level: extra spheres inserted between each adjacent base pair.

    Returns:
        List of int, length N_BODIES.
    """
    hl = np.asarray(HALF_LENGTHS)
    r = np.asarray(RADII)
    counts = []
    for i in range(N_BODIES):
        L = 2.0 * float(hl[i])
        R = float(r[i])
        n_base = max(1, round(L / (2.0 * R)))
        n_total = n_base + max(0, n_base - 1) * interpolation_level
        counts.append(n_total)
    return counts


def compute_max_human_spheres(interpolation_level=0, max_capsule_length=0.6, min_radius=0.05):
    """Max spheres a human capsule could decompose into (for static JAX shapes).

    Based on the longest possible human capsule (extended thigh/shin).
    """
    n_base = max(1, round(max_capsule_length / (2.0 * min_radius)))
    return n_base + max(0, n_base - 1) * interpolation_level


def sphere_centers(a, b, n):
    """Interpolate n sphere centers between capsule endpoints.

    Args:
        a: (3,) endpoint a (jnp).
        b: (3,) endpoint b (jnp).
        n: int, number of spheres (Python constant).

    Returns:
        (n, 3) jnp array of sphere centers.
    """
    if n == 1:
        return jnp.array([(a + b) / 2.0])
    t = jnp.linspace(0.0, 1.0, n)
    return (1.0 - t[:, None]) * b[None, :] + t[:, None] * a[None, :]


# ---------------------------------------------------------------------------
# Box mode helpers
# ---------------------------------------------------------------------------

# Box halfspace normals: 6 faces of an axis-aligned box (Ax <= b)
BOX_A = jnp.array([
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
])

# Pre-compute box b-vectors for each body: [radius, radius, half_length] repeated
BOX_B = jnp.array([
    [float(RADII[i]), float(RADII[i]), float(HALF_LENGTHS[i]),
     float(RADII[i]), float(RADII[i]), float(HALF_LENGTHS[i])]
    for i in range(N_BODIES)
])


def capsule_to_box_params(a, b, radius):
    """Convert capsule endpoints + radius to box center/rotation/b-vector.

    For human capsules (not in BODY_NAMES). Returns (center, rotation, b_vec).
    """
    center = (a + b) / 2.0
    axis = a - b
    length = jnp.linalg.norm(axis)
    half_length = length / 2.0 + radius

    # Build rotation aligning Z with capsule axis
    z_ax = axis / jnp.maximum(length, 1e-8)
    # Choose perpendicular axis
    up = jnp.array([0.0, 0.0, 1.0])
    dot = jnp.abs(jnp.dot(z_ax, up))
    up = jnp.where(dot > 0.999, jnp.array([1.0, 0.0, 0.0]), up)
    x_ax = jnp.cross(up, z_ax)
    x_ax = x_ax / jnp.linalg.norm(x_ax)
    y_ax = jnp.cross(z_ax, x_ax)
    rot = jnp.column_stack([x_ax, y_ax, z_ax])

    b_vec = jnp.array([radius, radius, half_length, radius, radius, half_length])
    return center, rot, b_vec
