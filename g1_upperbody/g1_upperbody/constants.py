"""Constants for G1 upper body + velocity RL policy.

All values extracted from the ONNX model metadata of the trained policy
(g1_velocity_upper_body/run1/policy.onnx).
"""

import numpy as np

# 29 joints in policy order (matches bridge motor index order 0-28).
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

NUM_JOINTS = 29
NUM_OBS = 106
NUM_ACTIONS = 29

# Home pose joint positions (from training keyframe).
DEFAULT_JOINT_POS = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,       # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,        # right leg
    0.0, 0.0, 0.0,                           # waist yaw/roll/pitch
    0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,  # left arm
    0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0, # right arm
], dtype=np.float32)

# Per-joint action scale: q_target = default_pos + action * action_scale.
# Computed from training config: 0.25 * effort_limit / stiffness per actuator.
ACTION_SCALE = np.array([
    0.548, 0.351, 0.548, 0.351, 0.439, 0.439,  # left leg
    0.548, 0.351, 0.548, 0.351, 0.439, 0.439,  # right leg
    0.548, 0.439, 0.439,                         # waist
    0.439, 0.439, 0.439, 0.439, 0.439, 0.075, 0.075,  # left arm
    0.439, 0.439, 0.439, 0.439, 0.439, 0.075, 0.075,  # right arm
], dtype=np.float32)

# The 8 upper body joints controlled by the command input.
UPPER_BODY_COMMAND_JOINTS = [
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
]

# Indices of upper body command joints within the 29-joint vector.
UPPER_BODY_COMMAND_INDICES = [
    JOINT_NAMES.index(j) for j in UPPER_BODY_COMMAND_JOINTS
]

# Gait phase period (seconds).
PHASE_PERIOD = 0.6

# Policy control dt (seconds).
POLICY_DT = 0.02
