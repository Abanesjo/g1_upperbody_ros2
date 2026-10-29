"""Humans walking between random waypoints, with a walking gait."""

import numpy as np

from g1_cbf.jax_kinematics import (
    CONTROLLED_JOINTS,
    CONTROLLED_JOINT_DEFAULTS,
    LEG_JOINTS,
)
from g1_human.human_body import PELVIS_HEIGHT, STANDING_KNEE_ANGLE

ARRIVAL_RADIUS = 0.3        # [m] a waypoint closer than this is reached
MAX_TURN_RATE = 1.5         # [rad/s]
ACCELERATION = 1.0          # [m/s^2]
STRIDE_LENGTH = 1.25        # [m] per gait cycle (two steps)

# Gait amplitudes at REFERENCE_SPEED; they scale with walking speed, so a
# human that stops returns to STANDING_POSE.
REFERENCE_SPEED = 1.0       # [m/s]
HIP_SWING = 0.35            # [rad] hip pitch amplitude
KNEE_SWING = 0.9            # [rad] extra knee flexion in mid-swing
ARM_SWING = 0.35            # [rad] shoulder pitch amplitude
ELBOW_SWING = 0.4           # [rad] extra elbow bend as the arm comes forward
WAIST_SWING = 0.1           # [rad] torso counter-rotation
PELVIS_BOB = 0.015          # [m] pelvis rise and fall, twice per stride

_J = CONTROLLED_JOINTS.index
_WAIST_YAW = _J('waist_yaw_joint')
_SHOULDER_PITCH = (_J('left_shoulder_pitch_joint'),
                   _J('right_shoulder_pitch_joint'))
_ELBOW = (_J('left_elbow_joint'), _J('right_elbow_joint'))
_HIP_PITCH = (LEG_JOINTS.index('left_hip_pitch_joint'),
              LEG_JOINTS.index('right_hip_pitch_joint'))

# Arms hanging by the sides, elbows slightly bent. The G1 elbow is bent 90
# degrees at 0 and close to straight at pi/2.
STANDING_POSE = CONTROLLED_JOINT_DEFAULTS.copy()
STANDING_POSE[list(_SHOULDER_PITCH)] = 0.0
STANDING_POSE[list(_ELBOW)] = 1.2


def _wrap(angles):
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


class CrowdSimulator:
    """Humans walking between random waypoints inside a square.

    Each human turns toward its waypoint at a limited rate and walks to it at
    its own speed, pauses there for a random time, then heads for a new
    waypoint with a new speed. Nobody leaves the square, and nobody starts
    within spawn_keepout_radius of its centre, where the robot starts.
    """

    def __init__(self, num_humans, side_length, center=(0.0, 0.0),
                 min_speed=0.4, max_speed=1.2, max_pause_sec=2.0,
                 spawn_keepout_radius=1.0, rng=None):
        if num_humans < 1:
            raise ValueError('num_humans must be at least 1')
        if side_length <= 0.0:
            raise ValueError('side_length must be positive')
        if not 0.0 < min_speed <= max_speed:
            raise ValueError('speeds must satisfy 0 < min_speed <= max_speed')
        if max_pause_sec < 0.0:
            raise ValueError('max_pause_sec must be non-negative')
        if not 0.0 <= spawn_keepout_radius < 0.5 * side_length:
            raise ValueError(
                'spawn_keepout_radius must be in [0, side_length / 2)'
            )

        self._rng = np.random.default_rng() if rng is None else rng
        self._center = np.asarray(center, dtype=np.float64)
        self._half = 0.5 * float(side_length)
        self._min_speed = float(min_speed)
        self._max_speed = float(max_speed)
        self._max_pause_sec = float(max_pause_sec)

        n = int(num_humans)
        self.position = self._spawn_positions(n, spawn_keepout_radius)
        self.heading = self._rng.uniform(-np.pi, np.pi, n)
        self.speed = np.zeros(n)
        self.target_speed = self._rng.uniform(min_speed, max_speed, n)
        self.waypoint = self._random_points(n)
        # Staggered starts, so the crowd does not set off in step.
        self.pause_left = self._rng.uniform(0.0, max_pause_sec, n)
        self.phase = self._rng.uniform(0.0, 2.0 * np.pi, n)

    @property
    def num_humans(self):
        return len(self.position)

    def step(self, dt):
        """Advance every human by dt seconds."""
        to_waypoint = self.waypoint - self.position
        arrived = (
            (self.pause_left <= 0.0)
            & (np.linalg.norm(to_waypoint, axis=1) < ARRIVAL_RADIUS)
        )
        count = int(np.count_nonzero(arrived))
        if count:
            self.pause_left[arrived] = self._rng.uniform(
                0.0, self._max_pause_sec, count,
            )
            self.waypoint[arrived] = self._random_points(count)
            self.target_speed[arrived] = self._rng.uniform(
                self._min_speed, self._max_speed, count,
            )
            to_waypoint = self.waypoint - self.position
        walking = self.pause_left <= 0.0
        self.pause_left = np.maximum(0.0, self.pause_left - dt)

        # Paused humans turn in place toward their next waypoint; walkers slow
        # down until they face it.
        bearing_error = _wrap(
            np.arctan2(to_waypoint[:, 1], to_waypoint[:, 0]) - self.heading
        )
        max_turn = MAX_TURN_RATE * dt
        self.heading = _wrap(
            self.heading + np.clip(bearing_error, -max_turn, max_turn)
        )
        goal_speed = np.where(
            walking,
            self.target_speed * np.clip(np.cos(bearing_error), 0.0, 1.0),
            0.0,
        )
        max_change = ACCELERATION * dt
        self.speed += np.clip(goal_speed - self.speed, -max_change, max_change)

        self.position += (self.speed * dt)[:, None] * np.column_stack([
            np.cos(self.heading), np.sin(self.heading),
        ])
        self.position = np.clip(
            self.position, self._center - self._half, self._center + self._half,
        )
        self.phase = (
            self.phase + 2.0 * np.pi * self.speed * dt / STRIDE_LENGTH
        ) % (2.0 * np.pi)

    def walking_pose(self):
        """Joint angles of the walking gait at the current phase and speed.

        Returns q (N, 11) in CONTROLLED_JOINTS order, q_legs (N, 6) in
        LEG_JOINTS order, knee_angles (N, 2) (left, right) and the pelvis
        height (N,).
        """
        n = self.num_humans
        scale = np.clip(self.speed / REFERENCE_SPEED, 0.0, 1.5)
        # The right leg runs half a cycle behind the left.
        leg_phase = np.column_stack([self.phase, self.phase + np.pi])
        swing = scale[:, None] * np.sin(leg_phase)

        q = np.tile(STANDING_POSE, (n, 1))
        # Each arm swings against the leg on its own side, and the elbow bends
        # (its angle falls) as the arm comes forward.
        q[:, list(_SHOULDER_PITCH)] += ARM_SWING * swing
        q[:, list(_ELBOW)] -= ELBOW_SWING * np.maximum(0.0, -swing)
        q[:, _WAIST_YAW] += WAIST_SWING * swing[:, 0]

        # Negative hip pitch moves the leg forward. The knee bends most in
        # mid-swing, while the leg passes under the body moving forward.
        q_legs = np.zeros((n, len(LEG_JOINTS)))
        q_legs[:, list(_HIP_PITCH)] = -HIP_SWING * swing
        knee_angles = STANDING_KNEE_ANGLE + KNEE_SWING * scale[:, None] * (
            np.maximum(0.0, np.cos(leg_phase))
        )
        pelvis_height = PELVIS_HEIGHT + PELVIS_BOB * scale * np.cos(
            2.0 * self.phase
        )
        return q, q_legs, knee_angles, pelvis_height

    def _random_points(self, count):
        return self._center + self._rng.uniform(
            -self._half, self._half, (count, 2),
        )

    def _spawn_positions(self, count, keepout_radius):
        positions = self._random_points(count)
        while True:
            too_close = (
                np.linalg.norm(positions - self._center, axis=1)
                < keepout_radius
            )
            if not too_close.any():
                return positions
            positions[too_close] = self._random_points(
                int(np.count_nonzero(too_close)),
            )
