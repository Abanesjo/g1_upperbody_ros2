#include "g1_bridge/low_cmd_builder.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "g1_bridge/gravity_compensation.hpp"

namespace {

using g1_bridge::BuildLowCommand;
using g1_bridge::ControlState;
using g1_bridge::kNumMotors;
using g1_bridge::kNumMotorSlots;
using g1_bridge::kUpperBodyStart;
using g1_bridge::LowCommand;
using g1_bridge::LowCommandInputs;

constexpr char G1_URDF[] = G1_BRIDGE_TEST_G1_URDF;
constexpr auto FRESH = std::chrono::milliseconds(1);
constexpr auto STALE = std::chrono::milliseconds(101);

const std::vector<std::string> JOINT_NAMES = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
};

const std::vector<double> JOINT_POSITIONS = {
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.37, 0.62, 0.0, 0.82, 0.0, 0.0, 0.0,
    0.33, -0.67, 0.0, 1.01, 0.0, 0.0, 0.0,
};

std::array<g1_bridge::PidGains, kNumMotors> UniformGains(double kp,
                                                         double kd) {
  std::array<g1_bridge::PidGains, kNumMotors> gains;
  gains.fill({kp, kd});
  return gains;
}

// What the bridge node feeds the builder: the safe command at the measured
// pose, and model gravity for the upper body of the real G1 URDF.
LowCommandInputs InputsAtRest(bool wbc, bool gravity) {
  LowCommandInputs inputs;
  inputs.gravity_enabled = g1_bridge::GravityFeedforwardEnabled(wbc, gravity);
  inputs.lowstate_age = FRESH;
  for (std::size_t i = 0; i < kNumMotors; ++i) {
    inputs.target_positions[i] = static_cast<float>(JOINT_POSITIONS[i]);
  }
  if (inputs.gravity_enabled) {
    g1_bridge::GravityCompensation model(G1_URDF, JOINT_NAMES);
    const auto &torques =
        model.Compute({1.0, 0.0, 0.0, 0.0}, JOINT_POSITIONS);
    for (std::size_t i = kUpperBodyStart; i < kNumMotors; ++i) {
      inputs.gravity_torques[i] = static_cast<float>(torques[i]);
    }
  }
  return inputs;
}

float MaxAbsUpperTorque(const LowCommand &command) {
  float max_abs = 0.0F;
  for (std::size_t i = kUpperBodyStart; i < kNumMotors; ++i) {
    max_abs = std::max(max_abs, std::abs(command.motor_cmd[i].tau));
  }
  return max_abs;
}

float MaxAbsLowerTorque(const LowCommand &command) {
  float max_abs = 0.0F;
  for (std::size_t i = 0; i < kUpperBodyStart; ++i) {
    max_abs = std::max(max_abs, std::abs(command.motor_cmd[i].tau));
  }
  return max_abs;
}

// CRC-32/MPEG-2 one byte at a time, written independently of Crc32Core.
std::uint32_t ReferenceCrc(const std::vector<std::uint8_t> &bytes) {
  std::uint32_t crc = 0xFFFFFFFF;
  for (const std::uint8_t byte : bytes) {
    crc ^= static_cast<std::uint32_t>(byte) << 24;
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x80000000) ? (crc << 1) ^ 0x04c11db7 : crc << 1;
    }
  }
  return crc;
}

// Unitree feeds its CRC 32-bit words, most significant byte first.
std::uint32_t ReferenceCrcOfWords(const std::vector<std::uint32_t> &words) {
  std::vector<std::uint8_t> bytes;
  for (const std::uint32_t word : words) {
    bytes.push_back(static_cast<std::uint8_t>(word >> 24));
    bytes.push_back(static_cast<std::uint8_t>(word >> 16));
    bytes.push_back(static_cast<std::uint8_t>(word >> 8));
    bytes.push_back(static_cast<std::uint8_t>(word));
  }
  return ReferenceCrc(bytes);
}

// The command laid out by hand as Unitree's C struct, without the crc word:
// two mode bytes and two padding bytes, 35 motor commands of 28 bytes (mode
// byte, three padding bytes, five floats, one reserve word), four reserve
// words.
std::vector<std::uint32_t> SerializeForCrc(const LowCommand &command) {
  constexpr std::size_t MOTOR_OFFSET = 4;
  constexpr std::size_t MOTOR_SIZE = 28;
  constexpr std::size_t RESERVE_OFFSET = MOTOR_OFFSET + 35 * MOTOR_SIZE;
  std::vector<std::uint8_t> bytes(RESERVE_OFFSET + 16, 0);

  bytes[0] = command.mode_pr;
  bytes[1] = command.mode_machine;
  for (std::size_t i = 0; i < kNumMotorSlots; ++i) {
    const auto &motor_cmd = command.motor_cmd[i];
    std::uint8_t *motor = bytes.data() + MOTOR_OFFSET + i * MOTOR_SIZE;
    motor[0] = motor_cmd.mode;
    std::memcpy(motor + 4, &motor_cmd.q, 4);
    std::memcpy(motor + 8, &motor_cmd.dq, 4);
    std::memcpy(motor + 12, &motor_cmd.tau, 4);
    std::memcpy(motor + 16, &motor_cmd.kp, 4);
    std::memcpy(motor + 20, &motor_cmd.kd, 4);
    std::memcpy(motor + 24, &motor_cmd.reserve, 4);
  }
  std::memcpy(bytes.data() + RESERVE_OFFSET, command.reserve.data(), 16);

  std::vector<std::uint32_t> words(bytes.size() / 4);
  std::memcpy(words.data(), bytes.data(), bytes.size());
  return words;
}

TEST(LowCmdBuilder, GravityFeedforwardModes) {
  auto inputs = InputsAtRest(false, true);
  const auto gains = UniformGains(100.0, 3.0);

  inputs.state = ControlState::kNeutral;
  auto command = BuildLowCommand(inputs, gains);
  EXPECT_FLOAT_EQ(command.motor_cmd[15].kp, 50.0F);
  EXPECT_GT(MaxAbsUpperTorque(command), 1e-3F);
  EXPECT_EQ(MaxAbsLowerTorque(command), 0.0F);

  // Gravity falls back to zero if fresh robot state stops arriving.
  inputs.lowstate_age = STALE;
  EXPECT_FALSE(g1_bridge::GravityFeedforwardApplies(inputs));
  command = BuildLowCommand(inputs, gains);
  EXPECT_FLOAT_EQ(command.motor_cmd[15].kp, 50.0F);
  EXPECT_EQ(MaxAbsUpperTorque(command), 0.0F);

  inputs.lowstate_age = FRESH;
  inputs.state = ControlState::kControl;
  command = BuildLowCommand(inputs, gains);
  EXPECT_FLOAT_EQ(command.motor_cmd[15].kp, 100.0F);
  EXPECT_GT(MaxAbsUpperTorque(command), 1e-3F);
  EXPECT_EQ(MaxAbsLowerTorque(command), 0.0F);

  inputs.state = ControlState::kDamp;
  EXPECT_FALSE(g1_bridge::GravityFeedforwardApplies(inputs));
  command = BuildLowCommand(inputs, gains);
  EXPECT_FLOAT_EQ(command.motor_cmd[15].kp, 0.0F);
  EXPECT_EQ(MaxAbsUpperTorque(command), 0.0F);
  EXPECT_EQ(MaxAbsLowerTorque(command), 0.0F);
}

TEST(LowCmdBuilder, WbcDisablesGravityFeedforward) {
  auto inputs = InputsAtRest(true, true);
  const auto gains = UniformGains(100.0, 3.0);
  EXPECT_FALSE(inputs.gravity_enabled);

  inputs.state = ControlState::kNeutral;
  auto command = BuildLowCommand(inputs, gains);
  EXPECT_FLOAT_EQ(command.motor_cmd[15].kp, 50.0F);
  EXPECT_EQ(MaxAbsUpperTorque(command), 0.0F);

  inputs.state = ControlState::kControl;
  command = BuildLowCommand(inputs, gains);
  EXPECT_FLOAT_EQ(command.motor_cmd[15].kp, 100.0F);
  EXPECT_EQ(MaxAbsUpperTorque(command), 0.0F);
}

TEST(LowCmdBuilder, GravityIsOffWithoutTheGravityFlag) {
  EXPECT_TRUE(g1_bridge::GravityFeedforwardEnabled(false, true));
  EXPECT_FALSE(g1_bridge::GravityFeedforwardEnabled(false, false));
  EXPECT_FALSE(g1_bridge::GravityFeedforwardEnabled(true, true));
  EXPECT_FALSE(g1_bridge::GravityFeedforwardEnabled(true, false));

  // Torques that are present but not enabled must never reach the command.
  LowCommandInputs inputs;
  inputs.gravity_torques.fill(5.0F);
  inputs.lowstate_age = FRESH;
  EXPECT_EQ(MaxAbsUpperTorque(BuildLowCommand(inputs, UniformGains(1, 1))),
            0.0F);
}

TEST(LowCmdBuilder, GravityUsesStateUpToTheTimeout) {
  LowCommandInputs inputs;
  inputs.gravity_enabled = true;
  inputs.gravity_torques.fill(2.0F);

  // No robot state at all counts as stale.
  EXPECT_FALSE(g1_bridge::GravityFeedforwardApplies(inputs));

  inputs.lowstate_age = g1_bridge::kGravityStateTimeout;
  EXPECT_TRUE(g1_bridge::GravityFeedforwardApplies(inputs));
  inputs.lowstate_age =
      g1_bridge::kGravityStateTimeout + std::chrono::microseconds(1);
  EXPECT_FALSE(g1_bridge::GravityFeedforwardApplies(inputs));
}

TEST(LowCmdBuilder, NeutralHoldsTargetsWithFixedGains) {
  LowCommandInputs inputs;
  inputs.state = ControlState::kNeutral;
  inputs.mode_pr = 1;
  inputs.mode_machine = 5;
  for (std::size_t i = 0; i < kNumMotors; ++i) {
    inputs.target_positions[i] = 0.01F * static_cast<float>(i + 1);
  }
  inputs.target_velocities.fill(0.4F);
  inputs.target_efforts.fill(0.5F);

  const auto command = BuildLowCommand(inputs, UniformGains(77.0, 7.0));

  EXPECT_EQ(command.mode_pr, 0);
  EXPECT_EQ(command.mode_machine, 5);
  for (std::size_t i = 0; i < kNumMotors; ++i) {
    const auto &motor_cmd = command.motor_cmd[i];
    EXPECT_EQ(motor_cmd.mode, 1) << i;
    EXPECT_FLOAT_EQ(motor_cmd.q, inputs.target_positions[i]) << i;
    EXPECT_EQ(motor_cmd.dq, 0.0F) << i;
    EXPECT_EQ(motor_cmd.tau, 0.0F) << i;
    EXPECT_FLOAT_EQ(motor_cmd.kp, i < 13 ? 100.0F : 50.0F) << i;
    EXPECT_FLOAT_EQ(motor_cmd.kd, 1.0F) << i;
  }
}

TEST(LowCmdBuilder, DampDropsStiffness) {
  LowCommandInputs inputs;
  inputs.state = ControlState::kDamp;
  inputs.mode_pr = 1;
  inputs.mode_machine = 5;
  inputs.target_positions.fill(0.3F);
  inputs.target_velocities.fill(0.4F);
  inputs.target_efforts.fill(0.5F);
  inputs.gravity_enabled = true;
  inputs.gravity_torques.fill(2.0F);
  inputs.lowstate_age = FRESH;

  const auto command = BuildLowCommand(inputs, UniformGains(77.0, 7.0));

  EXPECT_EQ(command.mode_pr, 0);
  EXPECT_EQ(command.mode_machine, 5);
  for (std::size_t i = 0; i < kNumMotors; ++i) {
    const auto &motor_cmd = command.motor_cmd[i];
    EXPECT_EQ(motor_cmd.mode, 1) << i;
    EXPECT_FLOAT_EQ(motor_cmd.q, 0.3F) << i;
    EXPECT_EQ(motor_cmd.dq, 0.0F) << i;
    EXPECT_EQ(motor_cmd.tau, 0.0F) << i;
    EXPECT_EQ(motor_cmd.kp, 0.0F) << i;
    EXPECT_FLOAT_EQ(motor_cmd.kd, 2.0F) << i;
  }
}

TEST(LowCmdBuilder, ControlForwardsTargetsWithConfiguredGains) {
  LowCommandInputs inputs;
  inputs.state = ControlState::kControl;
  inputs.mode_pr = 1;
  inputs.mode_machine = 5;
  inputs.target_positions.fill(0.3F);
  inputs.target_velocities.fill(0.4F);
  inputs.target_efforts.fill(0.5F);
  inputs.gravity_enabled = true;
  inputs.gravity_torques.fill(2.0F);
  inputs.lowstate_age = FRESH;

  std::array<g1_bridge::PidGains, kNumMotors> gains;
  for (std::size_t i = 0; i < kNumMotors; ++i) {
    gains[i] = {10.0 + static_cast<double>(i), 1.0 + static_cast<double>(i)};
  }

  const auto command = BuildLowCommand(inputs, gains);

  EXPECT_EQ(command.mode_pr, 1);
  EXPECT_EQ(command.mode_machine, 5);
  for (std::size_t i = 0; i < kNumMotors; ++i) {
    const auto &motor_cmd = command.motor_cmd[i];
    EXPECT_EQ(motor_cmd.mode, 1) << i;
    EXPECT_FLOAT_EQ(motor_cmd.q, 0.3F) << i;
    EXPECT_FLOAT_EQ(motor_cmd.dq, 0.4F) << i;
    // Gravity is added to the commanded effort, on the upper body only.
    EXPECT_FLOAT_EQ(motor_cmd.tau, i < kUpperBodyStart ? 0.5F : 2.5F) << i;
    EXPECT_FLOAT_EQ(motor_cmd.kp, 10.0F + static_cast<float>(i)) << i;
    EXPECT_FLOAT_EQ(motor_cmd.kd, 1.0F + static_cast<float>(i)) << i;
  }
}

TEST(LowCmdBuilder, LeavesUnusedMotorSlotsDisabled) {
  LowCommandInputs inputs;
  inputs.state = ControlState::kControl;
  inputs.target_positions.fill(0.3F);

  const auto command = BuildLowCommand(inputs, UniformGains(77.0, 7.0));

  for (std::size_t i = kNumMotors; i < kNumMotorSlots; ++i) {
    const auto &motor_cmd = command.motor_cmd[i];
    EXPECT_EQ(motor_cmd.mode, 0) << i;
    EXPECT_EQ(motor_cmd.q, 0.0F) << i;
    EXPECT_EQ(motor_cmd.kp, 0.0F) << i;
    EXPECT_EQ(motor_cmd.kd, 0.0F) << i;
  }
  for (const std::uint32_t word : command.reserve) {
    EXPECT_EQ(word, 0U);
  }
}

TEST(LowCmdCrc, ReferenceMatchesThePublishedCheckValue) {
  const std::string check = "123456789";
  EXPECT_EQ(ReferenceCrc({check.begin(), check.end()}), 0x0376E6E7U);
}

TEST(LowCmdCrc, CoreMatchesReferenceOnWords) {
  const std::vector<std::uint32_t> words = {
      0x31323334, 0x35363738, 0x00000000, 0xFFFFFFFF, 0xDEADBEEF, 0x00000001,
  };
  for (std::size_t count = 0; count <= words.size(); ++count) {
    EXPECT_EQ(g1_bridge::Crc32Core(words.data(), count),
              ReferenceCrcOfWords({words.begin(), words.begin() + count}))
        << count;
  }
}

TEST(LowCmdCrc, CoversTheUnitreeStructLayout) {
  LowCommand command;
  EXPECT_EQ(SerializeForCrc(command).size(), 250U);
  EXPECT_EQ(g1_bridge::ComputeLowCommandCrc(command),
            ReferenceCrcOfWords(SerializeForCrc(command)));

  command.mode_pr = 1;
  command.mode_machine = 5;
  for (std::size_t i = 0; i < kNumMotorSlots; ++i) {
    auto &motor_cmd = command.motor_cmd[i];
    const auto n = static_cast<float>(i + 1);
    motor_cmd.mode = static_cast<std::uint8_t>(i % 2);
    motor_cmd.q = 0.11F * n;
    motor_cmd.dq = -0.07F * n;
    motor_cmd.tau = 1.3F * n;
    motor_cmd.kp = 40.2F + n;
    motor_cmd.kd = 2.6F + n;
    motor_cmd.reserve = 0x01020304U * static_cast<std::uint32_t>(i + 1);
  }
  command.reserve = {0x11111111, 0x22222222, 0x33333333, 0x44444444};
  // The crc field itself is not part of the checksum.
  command.crc = 0xA5A5A5A5;

  EXPECT_EQ(g1_bridge::ComputeLowCommandCrc(command),
            ReferenceCrcOfWords(SerializeForCrc(command)));
}

TEST(LowCmdCrc, BuilderSignsTheCommand) {
  auto inputs = InputsAtRest(false, true);
  inputs.state = ControlState::kControl;
  inputs.mode_machine = 5;

  const auto command = BuildLowCommand(inputs, UniformGains(100.0, 3.0));
  EXPECT_EQ(command.crc, ReferenceCrcOfWords(SerializeForCrc(command)));

  inputs.target_positions[28] += 0.001F;
  const auto changed = BuildLowCommand(inputs, UniformGains(100.0, 3.0));
  EXPECT_NE(changed.crc, command.crc);
  EXPECT_EQ(changed.crc, ReferenceCrcOfWords(SerializeForCrc(changed)));
}

}  // namespace
