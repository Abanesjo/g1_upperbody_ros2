"""Stable wide-CSV schema shared by the logger and offline plotter."""

from dataclasses import dataclass


# These mirror g1_cbf.jax_kinematics without importing JAX into the live
# low-rate logger process. Package tests guard against drift.
CONTROLLED_JOINT_NAMES = (
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
)

CONTROLLED_JOINT_DEFAULTS = (
    0.0,
    0.0,
    0.0,
    0.35,
    0.18,
    0.0,
    0.87,
    0.35,
    -0.18,
    0.0,
    0.87,
)

LEG_JOINT_NAMES = (
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
)

JOINT_NAMES = CONTROLLED_JOINT_NAMES + LEG_JOINT_NAMES
JOINT_DEFAULTS = dict(zip(
    JOINT_NAMES,
    CONTROLLED_JOINT_DEFAULTS + (0.0,) * len(LEG_JOINT_NAMES),
))
N_HUMAN_CAPSULES = 11


@dataclass(frozen=True)
class JointSample:
    """CBF joint vector and which entries actually appeared in a message."""

    values: dict
    observed: dict
    observed_count: int


def extract_joint_state(names, positions):
    """Extract the CBF joints with the same defaults used by ``g1_cbf``."""
    received = dict(zip(names, positions))
    values = {}
    observed = {}
    for name in JOINT_NAMES:
        is_observed = name in received
        observed[name] = is_observed
        values[name] = float(
            received[name] if is_observed else JOINT_DEFAULTS[name]
        )
    return JointSample(
        values=values,
        observed=observed,
        observed_count=sum(observed.values()),
    )


def stamp_to_sec(stamp):
    """Convert a ROS builtin_interfaces/Time-like value to float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def joint_value_field(name):
    return f'joint__{name}'


def joint_observed_field(name):
    return f'joint__{name}__observed'


def human_slot_fields(index):
    prefix = f'human_{int(index):02d}'
    return (
        f'{prefix}_present',
        f'{prefix}_fresh',
        f'{prefix}_name',
        f'{prefix}_a_x_m',
        f'{prefix}_a_y_m',
        f'{prefix}_a_z_m',
        f'{prefix}_b_x_m',
        f'{prefix}_b_y_m',
        f'{prefix}_b_z_m',
        f'{prefix}_radius_m',
    )


BASE_FIELDS = (
    'sample_index',
    'ros_time_sec',
    'elapsed_sec',
    'joint_state_valid',
    'joint_state_stamp_sec',
    'joint_state_age_sec',
    'joint_state_observed_count',
)

JOINT_FIELDS = tuple(
    field
    for name in JOINT_NAMES
    for field in (joint_value_field(name), joint_observed_field(name))
)

HUMAN_METADATA_FIELDS = (
    'human_valid',
    'human_stamp_sec',
    'human_age_sec',
    'human_source_frame',
    'human_frame_id',
    'human_received_count',
    'human_used_count',
)

HUMAN_FIELDS = tuple(
    field
    for index in range(N_HUMAN_CAPSULES)
    for field in human_slot_fields(index)
)

ODOM_FIELDS = (
    'odom_valid',
    'odom_stamp_sec',
    'odom_age_sec',
    'odom_frame_id',
    'odom_child_frame_id',
    'odom_x_m',
    'odom_y_m',
)

GATE_FIELDS = (
    'cbf_enabled',
    'external_enabled',
)

WORKSPACE_FIELDS = (
    'workspace_valid',
    'workspace_stamp_sec',
    'workspace_age_sec',
    'workspace_frame_id',
    'workspace_child_frame_id',
    'workspace_enabled',
    'workspace_capture_pending',
    'workspace_generation',
    'workspace_center_x_m',
    'workspace_center_y_m',
    'workspace_center_z_m',
    'workspace_qx',
    'workspace_qy',
    'workspace_qz',
    'workspace_qw',
    'workspace_radius_m',
    'workspace_activation_valid',
    'workspace_activation_generation',
    'workspace_activation_stamp_sec',
    'workspace_activation_frame_id',
    'workspace_activation_child_frame_id',
    'workspace_activation_center_x_m',
    'workspace_activation_center_y_m',
    'workspace_activation_qx',
    'workspace_activation_qy',
    'workspace_activation_qz',
    'workspace_activation_qw',
    'workspace_path_valid',
    'workspace_path_stamp_sec',
    'workspace_path_age_sec',
    'workspace_path_frame_id',
    'workspace_path_sequence',
    'workspace_path_point_count',
    'workspace_path_xy_json',
)

TF_FIELDS = (
    'tf_valid',
    'tf_stamp_sec',
    'tf_age_sec',
    'tf_world_frame',
    'tf_pelvis_frame',
    'tf_world_pelvis_tx_m',
    'tf_world_pelvis_ty_m',
    'tf_world_pelvis_tz_m',
    'tf_world_pelvis_qx',
    'tf_world_pelvis_qy',
    'tf_world_pelvis_qz',
    'tf_world_pelvis_qw',
)

CONFIG_FIELDS = (
    'collision_geometry',
    'internal_margin_phi_m',
    'external_margin_phi_m',
    'external_torso_margin_phi_m',
    'human_timeout_sec',
    'tf_stale_timeout_sec',
)

CSV_FIELDS = (
    BASE_FIELDS
    + JOINT_FIELDS
    + HUMAN_METADATA_FIELDS
    + HUMAN_FIELDS
    + ODOM_FIELDS
    + GATE_FIELDS
    + WORKSPACE_FIELDS
    + TF_FIELDS
    + CONFIG_FIELDS
)
