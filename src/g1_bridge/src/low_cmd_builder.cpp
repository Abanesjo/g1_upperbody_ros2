#include "g1_bridge/low_cmd_builder.hpp"

#include <cstring>

namespace g1_bridge {

namespace {

constexpr std::uint8_t MODE_PR = 0;
constexpr float NEUTRAL_LEG_KP = 100.0F;
constexpr float NEUTRAL_UPPER_KP = 50.0F;
constexpr float NEUTRAL_KD = 1.0F;
constexpr float DAMP_KP = 0.0F;
constexpr float DAMP_KD = 2.0F;

// Layout of Unitree's LowCmd as the robot checksums it.
struct RawMotorCmd {
  std::uint8_t mode;
  float q;
  float dq;
  float tau;
  float Kp;
  float Kd;
  std::uint32_t reserve;
};

struct RawLowCmd {
  std::uint8_t modePr;
  std::uint8_t modeMachine;
  std::array<RawMotorCmd, kNumMotorSlots> motorCmd;
  std::array<std::uint32_t, 4> reserve;
  std::uint32_t crc;
};

static_assert(sizeof(RawMotorCmd) == 28, "unexpected MotorCmd layout");
static_assert(sizeof(RawLowCmd) == 1004, "unexpected LowCmd layout");

constexpr std::size_t RAW_LOW_CMD_WORDS = sizeof(RawLowCmd) / 4;

}  // namespace

bool GravityFeedforwardApplies(const LowCommandInputs &inputs) {
  return inputs.gravity_enabled && inputs.state != ControlState::kDamp &&
         inputs.lowstate_age <= kGravityStateTimeout;
}

LowCommand BuildLowCommand(
    const LowCommandInputs &inputs,
    const std::array<PidGains, kNumMotors> &control_gains) {
  LowCommand cmd;
  cmd.mode_pr = inputs.state == ControlState::kControl ? inputs.mode_pr
                                                       : MODE_PR;
  cmd.mode_machine = inputs.mode_machine;

  for (std::size_t i = 0; i < kNumMotors; ++i) {
    auto &motor_cmd = cmd.motor_cmd[i];
    motor_cmd.mode = 1;
    motor_cmd.q = inputs.target_positions[i];
    switch (inputs.state) {
      case ControlState::kDamp:
        motor_cmd.kp = DAMP_KP;
        motor_cmd.kd = DAMP_KD;
        break;
      case ControlState::kControl:
        motor_cmd.dq = inputs.target_velocities[i];
        motor_cmd.tau = inputs.target_efforts[i];
        motor_cmd.kp = static_cast<float>(control_gains[i].kp);
        motor_cmd.kd = static_cast<float>(control_gains[i].kd);
        break;
      case ControlState::kNeutral:
        motor_cmd.kp = (i < 13) ? NEUTRAL_LEG_KP : NEUTRAL_UPPER_KP;
        motor_cmd.kd = NEUTRAL_KD;
        break;
    }
  }

  if (GravityFeedforwardApplies(inputs)) {
    for (std::size_t i = kUpperBodyStart; i < kNumMotors; ++i) {
      cmd.motor_cmd[i].tau += inputs.gravity_torques[i];
    }
  }

  cmd.crc = ComputeLowCommandCrc(cmd);
  return cmd;
}

std::uint32_t Crc32Core(const std::uint32_t *words, std::size_t count) {
  std::uint32_t xbit = 0;
  std::uint32_t data = 0;
  std::uint32_t CRC32 = 0xFFFFFFFF;
  const std::uint32_t dwPolynomial = 0x04c11db7;
  for (std::size_t i = 0; i < count; i++) {
    xbit = 1U << 31;
    data = words[i];
    for (std::uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else {
        CRC32 <<= 1;
      }
      if (data & xbit) {
        CRC32 ^= dwPolynomial;
      }

      xbit >>= 1;
    }
  }
  return CRC32;
}

std::uint32_t ComputeLowCommandCrc(const LowCommand &command) {
  RawLowCmd raw;
  // The padding bytes are part of the checksum, so they have to be zero.
  std::memset(&raw, 0, sizeof(raw));

  raw.modePr = command.mode_pr;
  raw.modeMachine = command.mode_machine;
  for (std::size_t i = 0; i < kNumMotorSlots; ++i) {
    raw.motorCmd[i].mode = command.motor_cmd[i].mode;
    raw.motorCmd[i].q = command.motor_cmd[i].q;
    raw.motorCmd[i].dq = command.motor_cmd[i].dq;
    raw.motorCmd[i].tau = command.motor_cmd[i].tau;
    raw.motorCmd[i].Kp = command.motor_cmd[i].kp;
    raw.motorCmd[i].Kd = command.motor_cmd[i].kd;
    raw.motorCmd[i].reserve = command.motor_cmd[i].reserve;
  }
  raw.reserve = command.reserve;

  std::array<std::uint32_t, RAW_LOW_CMD_WORDS> words;
  std::memcpy(words.data(), &raw, sizeof(raw));
  return Crc32Core(words.data(), RAW_LOW_CMD_WORDS - 1);
}

}  // namespace g1_bridge
