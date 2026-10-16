#include "g1_bridge/unitree_link.hpp"

#include <net/if.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/ros2/PoseStamped_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// rosgraph_msgs/Clock is not part of the SDK; this is the type unitree_mujoco
// publishes on rt/clock.
#include "Clock_.hpp"

namespace g1_bridge {

namespace {

using geometry_msgs::msg::dds_::PoseStamped_;
using rosgraph_msgs::msg::dds_::Clock_;
using unitree_hg::msg::dds_::LowCmd_;
using unitree_hg::msg::dds_::LowState_;

constexpr char LOW_CMD_TOPIC[] = "rt/lowcmd";
constexpr char LOW_STATE_TOPIC[] = "rt/lowstate";
constexpr char CLOCK_TOPIC[] = "rt/clock";
constexpr char PELVIS_POSE_TOPIC[] = "rt/pose/pelvis";
constexpr int MAX_DOMAIN_ID = 232;
constexpr float MOTION_CALL_TIMEOUT_SEC = 5.0F;
// Each subscription hands its messages to a thread of its own and keeps only
// the newest one, so a slow callback never stalls the DDS receive thread.
constexpr int READER_QUEUE_LENGTH = 1;

}  // namespace

class UnitreeLink::Impl {
 public:
  Impl(int domain_id, const std::string &network_interface) {
    if (domain_id < 0 || domain_id > MAX_DOMAIN_ID) {
      throw std::invalid_argument(
          "domain_id must be between 0 and " + std::to_string(MAX_DOMAIN_ID));
    }
    // An empty name would let CycloneDDS pick an interface by itself.
    if (network_interface.empty()) {
      throw std::invalid_argument("network_interface must be set");
    }
    if (if_nametoindex(network_interface.c_str()) == 0) {
      throw std::runtime_error(
          "network interface '" + network_interface + "' does not exist");
    }

    unitree::robot::ChannelFactory::Instance()->Init(domain_id,
                                                     network_interface);
  }

  void EnableLowCommand() {
    if (low_cmd_publisher_) {
      return;
    }
    low_cmd_publisher_ =
        std::make_unique<unitree::robot::ChannelPublisher<LowCmd_>>(
            LOW_CMD_TOPIC);
    low_cmd_publisher_->InitChannel();
  }

  void SubscribeRobotState(RobotStateCallback callback) {
    robot_state_callback_ = std::move(callback);
    low_state_subscriber_ =
        std::make_unique<unitree::robot::ChannelSubscriber<LowState_>>(
            LOW_STATE_TOPIC);
    low_state_subscriber_->InitChannel(
        [this](const void *message) { OnLowState(message); },
        READER_QUEUE_LENGTH);
  }

  void SubscribeClock(ClockCallback callback) {
    clock_callback_ = std::move(callback);
    clock_subscriber_ =
        std::make_unique<unitree::robot::ChannelSubscriber<Clock_>>(
            CLOCK_TOPIC);
    clock_subscriber_->InitChannel(
        [this](const void *message) { OnClock(message); },
        READER_QUEUE_LENGTH);
  }

  void SubscribePelvisPose(PoseCallback callback) {
    pelvis_pose_callback_ = std::move(callback);
    pelvis_pose_subscriber_ =
        std::make_unique<unitree::robot::ChannelSubscriber<PoseStamped_>>(
            PELVIS_POSE_TOPIC);
    pelvis_pose_subscriber_->InitChannel(
        [this](const void *message) { OnPelvisPose(message); },
        READER_QUEUE_LENGTH);
  }

  bool WriteLowCommand(const LowCommand &command) {
    if (!low_cmd_publisher_) {
      return false;
    }
    LowCmd_ msg;
    msg.mode_pr() = command.mode_pr;
    msg.mode_machine() = command.mode_machine;
    for (std::size_t i = 0; i < kNumMotorSlots; ++i) {
      const auto &motor_cmd = command.motor_cmd[i];
      auto &motor_msg = msg.motor_cmd()[i];
      motor_msg.mode() = motor_cmd.mode;
      motor_msg.q() = motor_cmd.q;
      motor_msg.dq() = motor_cmd.dq;
      motor_msg.tau() = motor_cmd.tau;
      motor_msg.kp() = motor_cmd.kp;
      motor_msg.kd() = motor_cmd.kd;
      motor_msg.reserve() = motor_cmd.reserve;
    }
    msg.reserve() = command.reserve;
    msg.crc() = command.crc;
    return low_cmd_publisher_->Write(msg);
  }

  std::int32_t CheckMotionMode(std::string &form, std::string &name) {
    return MotionSwitcher().CheckMode(form, name);
  }

  std::int32_t ReleaseMotionMode() { return MotionSwitcher().ReleaseMode(); }

 private:
  unitree::robot::b2::MotionSwitcherClient &MotionSwitcher() {
    if (!motion_switcher_) {
      motion_switcher_ =
          std::make_unique<unitree::robot::b2::MotionSwitcherClient>();
      motion_switcher_->SetTimeout(MOTION_CALL_TIMEOUT_SEC);
      motion_switcher_->Init();
    }
    return *motion_switcher_;
  }

  void OnLowState(const void *message) {
    const auto &msg = *static_cast<const LowState_ *>(message);

    RobotState state;
    state.mode_pr = msg.mode_pr();
    state.mode_machine = msg.mode_machine();
    state.tick = msg.tick();
    state.quaternion_wxyz = msg.imu_state().quaternion();
    state.gyroscope = msg.imu_state().gyroscope();
    state.accelerometer = msg.imu_state().accelerometer();
    for (std::size_t i = 0; i < kNumMotorSlots; ++i) {
      const auto &motor = msg.motor_state()[i];
      state.q[i] = motor.q();
      state.dq[i] = motor.dq();
      state.tau_est[i] = motor.tau_est();
    }
    robot_state_callback_(state);
  }

  void OnClock(const void *message) {
    const auto &msg = *static_cast<const Clock_ *>(message);
    clock_callback_({msg.clock().sec(), msg.clock().nanosec()});
  }

  void OnPelvisPose(const void *message) {
    const auto &msg = *static_cast<const PoseStamped_ *>(message);

    StampedPose pose;
    pose.stamp = {msg.header().stamp().sec(), msg.header().stamp().nanosec()};
    pose.frame_id = msg.header().frame_id();
    pose.position = {msg.pose().position().x(), msg.pose().position().y(),
                     msg.pose().position().z()};
    pose.orientation_xyzw = {
        msg.pose().orientation().x(), msg.pose().orientation().y(),
        msg.pose().orientation().z(), msg.pose().orientation().w()};
    pelvis_pose_callback_(pose);
  }

  // The callbacks are declared first so that they outlive the subscribers
  // that call them.
  RobotStateCallback robot_state_callback_;
  ClockCallback clock_callback_;
  PoseCallback pelvis_pose_callback_;
  std::unique_ptr<unitree::robot::ChannelPublisher<LowCmd_>>
      low_cmd_publisher_;
  std::unique_ptr<unitree::robot::ChannelSubscriber<LowState_>>
      low_state_subscriber_;
  std::unique_ptr<unitree::robot::ChannelSubscriber<Clock_>> clock_subscriber_;
  std::unique_ptr<unitree::robot::ChannelSubscriber<PoseStamped_>>
      pelvis_pose_subscriber_;
  std::unique_ptr<unitree::robot::b2::MotionSwitcherClient> motion_switcher_;
};

UnitreeLink::UnitreeLink(int domain_id, const std::string &network_interface)
    : impl_(std::make_unique<Impl>(domain_id, network_interface)) {}

UnitreeLink::~UnitreeLink() = default;

void UnitreeLink::SubscribeRobotState(RobotStateCallback callback) {
  impl_->SubscribeRobotState(std::move(callback));
}

void UnitreeLink::SubscribeClock(ClockCallback callback) {
  impl_->SubscribeClock(std::move(callback));
}

void UnitreeLink::SubscribePelvisPose(PoseCallback callback) {
  impl_->SubscribePelvisPose(std::move(callback));
}

void UnitreeLink::EnableLowCommand() { impl_->EnableLowCommand(); }

bool UnitreeLink::WriteLowCommand(const LowCommand &command) {
  return impl_->WriteLowCommand(command);
}

std::int32_t UnitreeLink::CheckMotionMode(std::string &form,
                                          std::string &name) {
  return impl_->CheckMotionMode(form, name);
}

std::int32_t UnitreeLink::ReleaseMotionMode() {
  return impl_->ReleaseMotionMode();
}

}  // namespace g1_bridge
