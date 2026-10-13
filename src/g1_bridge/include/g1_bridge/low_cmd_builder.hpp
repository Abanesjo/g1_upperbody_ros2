#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace g1_bridge {

inline constexpr std::size_t kNumMotors = 29;
// unitree_hg LowCmd and LowState carry 35 motor slots; the 29-DOF G1 uses the
// first 29.
inline constexpr std::size_t kNumMotorSlots = 35;
inline constexpr std::size_t kUpperBodyStart = 12;
inline constexpr std::chrono::milliseconds kGravityStateTimeout{100};

// Plain mirrors of unitree_hg MotorCmd and LowCmd, so that building a command
// needs neither ROS nor the Unitree SDK.
struct MotorCommand {
  std::uint8_t mode{};
  float q{};
  float dq{};
  float tau{};
  float kp{};
  float kd{};
  std::uint32_t reserve{};
};

struct LowCommand {
  std::uint8_t mode_pr{};
  std::uint8_t mode_machine{};
  std::array<MotorCommand, kNumMotorSlots> motor_cmd{};
  std::array<std::uint32_t, 4> reserve{};
  std::uint32_t crc{};
};

enum class ControlState {
  kNeutral,
  kControl,
  kDamp,
};

struct PidGains {
  double kp{};
  double kd{};
};

using JointArray = std::array<float, kNumMotors>;

struct LowCommandInputs {
  ControlState state{ControlState::kNeutral};
  // Echoed from the latest LowState.
  std::uint8_t mode_pr{};
  std::uint8_t mode_machine{};
  JointArray target_positions{};
  JointArray target_velocities{};
  JointArray target_efforts{};
  JointArray gravity_torques{};
  bool gravity_enabled{false};
  // Time since the LowState that gravity_torques were computed from.
  std::chrono::steady_clock::duration lowstate_age{
      std::chrono::steady_clock::duration::max()};
};

// Gravity feed-forward belongs to the classical upper-body PD path. With wbc
// the policy drives every joint and the bridge adds nothing.
constexpr bool GravityFeedforwardEnabled(bool wbc, bool gravity) {
  return !wbc && gravity;
}

// True when BuildLowCommand adds gravity_torques for these inputs: gravity is
// enabled, the state is neutral or control, and the robot state is fresh.
bool GravityFeedforwardApplies(const LowCommandInputs &inputs);

// Builds the command for one 2 ms tick, CRC included. control_gains are used
// in the control state only; neutral and damp use fixed gains.
LowCommand BuildLowCommand(
    const LowCommandInputs &inputs,
    const std::array<PidGains, kNumMotors> &control_gains);

// Unitree's CRC-32 (polynomial 0x04c11db7, most significant bit first) over
// 32-bit words.
std::uint32_t Crc32Core(const std::uint32_t *words, std::size_t count);

// The checksum the robot expects in LowCommand::crc: Crc32Core over the
// command laid out as Unitree's C struct, without its trailing crc word.
std::uint32_t ComputeLowCommandCrc(const LowCommand &command);

}  // namespace g1_bridge
