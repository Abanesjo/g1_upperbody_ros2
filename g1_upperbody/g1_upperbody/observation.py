"""Observation vector construction for the G1 upper body RL policy.

Builds the 106-dim observation expected by the ONNX policy:
  base_ang_vel (3) | projected_gravity (3) | command (3) | phase (2) |
  joint_pos (29) | joint_vel (29) | actions (29) | upper_body_command (8)
"""

import math
import numpy as np

from .constants import DEFAULT_JOINT_POS, NUM_OBS, NUM_ACTIONS, PHASE_PERIOD


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector *v* by the inverse of quaternion *q* (w, x, y, z).

    Transforms *v* from world frame into the body frame described by *q*.
    Matches the C++ deploy: ``projected_gravity_b = q.conjugate() * gravity``.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    # Conjugate: negate imaginary part to get inverse rotation
    u = np.array([-x, -y, -z])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


class ObservationBuilder:
    """Incrementally constructs 106-dim observation vectors."""

    def __init__(self, phase_period: float = PHASE_PERIOD):
        self.phase_period = phase_period
        self.global_phase = 0.0
        self.last_actions = np.zeros(NUM_ACTIONS, dtype=np.float32)

    def reset(self):
        self.global_phase = 0.0
        self.last_actions = np.zeros(NUM_ACTIONS, dtype=np.float32)

    def build(
        self,
        imu_ang_vel: np.ndarray,
        imu_quat: np.ndarray,
        cmd_vel: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        upper_body_cmd: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Construct the 106-dim observation vector.

        Args:
            imu_ang_vel: (3,) gyroscope reading [x, y, z] rad/s.
            imu_quat: (4,) quaternion [w, x, y, z] from IMU.
            cmd_vel: (3,) velocity command [lin_x, lin_y, ang_z].
            joint_pos: (29,) current joint positions in rad.
            joint_vel: (29,) current joint velocities in rad/s.
            upper_body_cmd: (8,) absolute target positions for 8 joints.
            dt: timestep in seconds.

        Returns:
            obs: (106,) observation vector.
        """
        obs = np.zeros(NUM_OBS, dtype=np.float32)
        idx = 0

        # 1. base_ang_vel (3)
        obs[idx:idx + 3] = imu_ang_vel
        idx += 3

        # 2. projected_gravity (3)
        gravity_world = np.array([0.0, 0.0, -1.0])
        obs[idx:idx + 3] = quat_rotate_inverse(imu_quat, gravity_world)
        idx += 3

        # 3. velocity command (3)
        obs[idx:idx + 3] = cmd_vel
        idx += 3

        # 4. gait phase (2): sin/cos, zeroed when standing
        self.global_phase = (self.global_phase + dt / self.phase_period) % 1.0
        cmd_norm = float(np.linalg.norm(cmd_vel))
        if cmd_norm >= 0.1:
            angle = self.global_phase * 2.0 * math.pi
            obs[idx] = math.sin(angle)
            obs[idx + 1] = math.cos(angle)
        idx += 2

        # 5. joint_pos relative to default (29)
        obs[idx:idx + 29] = joint_pos - DEFAULT_JOINT_POS
        idx += 29

        # 6. joint_vel (29)
        obs[idx:idx + 29] = joint_vel
        idx += 29

        # 7. last actions (29)
        obs[idx:idx + 29] = self.last_actions
        idx += 29

        # 8. upper_body_command (8)
        obs[idx:idx + 8] = upper_body_cmd
        idx += 8

        return obs

    def update_last_actions(self, actions: np.ndarray):
        self.last_actions = actions.copy()
