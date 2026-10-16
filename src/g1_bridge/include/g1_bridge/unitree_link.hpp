#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "g1_bridge/low_cmd_builder.hpp"

namespace g1_bridge {

// The fields of unitree_hg LowState that the bridge uses.
struct RobotState {
  std::uint8_t mode_pr{};
  std::uint8_t mode_machine{};
  std::uint32_t tick{};
  std::array<float, 4> quaternion_wxyz{};
  std::array<float, 3> gyroscope{};
  std::array<float, 3> accelerometer{};
  std::array<float, kNumMotorSlots> q{};
  std::array<float, kNumMotorSlots> dq{};
  std::array<float, kNumMotorSlots> tau_est{};
};

struct TimeStamp {
  std::int32_t sec{};
  std::uint32_t nanosec{};
};

struct StampedPose {
  TimeStamp stamp;
  std::string frame_id;
  std::array<double, 3> position{};
  std::array<double, 4> orientation_xyzw{0.0, 0.0, 0.0, 1.0};
};

// The robot side of the bridge: talks to the G1, or to unitree_mujoco, with
// unitree_sdk2 on one network interface. The SDK and its CycloneDDS headers
// are intentionally hidden so users of this header do not depend on them.
//
// Unitree's ChannelFactory is a process-wide singleton, so create at most one
// UnitreeLink per process. Callbacks run on SDK threads.
class UnitreeLink {
 public:
  using RobotStateCallback = std::function<void(const RobotState &)>;
  using ClockCallback = std::function<void(const TimeStamp &)>;
  using PoseCallback = std::function<void(const StampedPose &)>;

  // Throws if network_interface is empty or does not exist.
  UnitreeLink(int domain_id, const std::string &network_interface);
  ~UnitreeLink();

  UnitreeLink(const UnitreeLink &) = delete;
  UnitreeLink &operator=(const UnitreeLink &) = delete;

  // rt/lowstate
  void SubscribeRobotState(RobotStateCallback callback);
  // rt/clock and rt/pose/pelvis, which only unitree_mujoco publishes.
  void SubscribeClock(ClockCallback callback);
  void SubscribePelvisPose(PoseCallback callback);

  // rt/lowcmd. No writer exists until EnableLowCommand(), so a process that
  // only listens, like sim_relay_node, never appears as a lowcmd source.
  // WriteLowCommand returns false before that.
  void EnableLowCommand();
  bool WriteLowCommand(const LowCommand &command);

  // Unitree's motion switcher service, which exists on the robot only. Both
  // return the SDK error code, 0 on success. An empty name means that no
  // built-in motion service is active.
  std::int32_t CheckMotionMode(std::string &form, std::string &name);
  std::int32_t ReleaseMotionMode();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace g1_bridge
