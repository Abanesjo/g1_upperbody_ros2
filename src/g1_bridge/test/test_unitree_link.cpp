#include "g1_bridge/unitree_link.hpp"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/ros2/PoseStamped_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "Clock_.hpp"

namespace {

using geometry_msgs::msg::dds_::PoseStamped_;
using rosgraph_msgs::msg::dds_::Clock_;
using unitree_hg::msg::dds_::LowCmd_;
using unitree_hg::msg::dds_::LowState_;

// Not domain 0, which the robot and unitree_mujoco use: this test must never
// exchange messages with either.
constexpr int TEST_DOMAIN_ID = 73;
constexpr char TEST_INTERFACE[] = "lo";
constexpr auto RECEIVE_TIMEOUT = std::chrono::seconds(10);
constexpr auto PUBLISH_PERIOD = std::chrono::milliseconds(20);

// The robot checksums (sizeof(LowCmd) >> 2) - 1 words of its struct, so the
// SDK type has to be as large as the layout that ComputeLowCommandCrc covers.
static_assert(sizeof(LowCmd_) == 1004, "unexpected unitree_hg LowCmd_ layout");

// Holds the latest message that an SDK thread delivered.
template <typename T>
class Mailbox {
 public:
  void Put(const T &value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      value_ = value;
    }
    condition_.notify_all();
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    value_.reset();
  }

  // Calls publish every PUBLISH_PERIOD until a message accepted by matches
  // arrives.
  std::optional<T> WaitFor(const std::function<void()> &publish,
                           const std::function<bool(const T &)> &matches) {
    const auto deadline = std::chrono::steady_clock::now() + RECEIVE_TIMEOUT;
    std::unique_lock<std::mutex> lock(mutex_);
    while (std::chrono::steady_clock::now() < deadline) {
      if (value_ && matches(*value_)) {
        return value_;
      }
      lock.unlock();
      publish();
      lock.lock();
      condition_.wait_for(lock, PUBLISH_PERIOD);
    }
    return std::nullopt;
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<T> value_;
};

// One link and one in-process stand-in for the robot (or unitree_mujoco) for
// the whole test program, because the SDK's ChannelFactory can be initialised
// only once per process.
class UnitreeLinkTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    link_ = std::make_unique<g1_bridge::UnitreeLink>(TEST_DOMAIN_ID,
                                                     TEST_INTERFACE);
    wrote_before_enable_ = link_->WriteLowCommand(g1_bridge::LowCommand{});
    link_->EnableLowCommand();

    low_state_publisher_ =
        std::make_unique<unitree::robot::ChannelPublisher<LowState_>>(
            "rt/lowstate");
    low_state_publisher_->InitChannel();
    clock_publisher_ =
        std::make_unique<unitree::robot::ChannelPublisher<Clock_>>("rt/clock");
    clock_publisher_->InitChannel();
    pelvis_pose_publisher_ =
        std::make_unique<unitree::robot::ChannelPublisher<PoseStamped_>>(
            "rt/pose/pelvis");
    pelvis_pose_publisher_->InitChannel();
    low_cmd_subscriber_ =
        std::make_unique<unitree::robot::ChannelSubscriber<LowCmd_>>(
            "rt/lowcmd");
    low_cmd_subscriber_->InitChannel(
        [](const void *message) {
          low_cmds_.Put(*static_cast<const LowCmd_ *>(message));
        },
        1);

    link_->SubscribeRobotState(
        [](const g1_bridge::RobotState &state) { robot_states_.Put(state); });
    link_->SubscribeClock(
        [](const g1_bridge::TimeStamp &stamp) { clocks_.Put(stamp); });
    link_->SubscribePelvisPose(
        [](const g1_bridge::StampedPose &pose) { pelvis_poses_.Put(pose); });
  }

  static void TearDownTestSuite() {
    link_.reset();
    low_cmd_subscriber_.reset();
    pelvis_pose_publisher_.reset();
    clock_publisher_.reset();
    low_state_publisher_.reset();
  }

  static std::unique_ptr<g1_bridge::UnitreeLink> link_;
  static bool wrote_before_enable_;
  static std::unique_ptr<unitree::robot::ChannelPublisher<LowState_>>
      low_state_publisher_;
  static std::unique_ptr<unitree::robot::ChannelPublisher<Clock_>>
      clock_publisher_;
  static std::unique_ptr<unitree::robot::ChannelPublisher<PoseStamped_>>
      pelvis_pose_publisher_;
  static std::unique_ptr<unitree::robot::ChannelSubscriber<LowCmd_>>
      low_cmd_subscriber_;
  static Mailbox<g1_bridge::RobotState> robot_states_;
  static Mailbox<g1_bridge::TimeStamp> clocks_;
  static Mailbox<g1_bridge::StampedPose> pelvis_poses_;
  static Mailbox<LowCmd_> low_cmds_;
};

std::unique_ptr<g1_bridge::UnitreeLink> UnitreeLinkTest::link_;
bool UnitreeLinkTest::wrote_before_enable_ = true;
std::unique_ptr<unitree::robot::ChannelPublisher<LowState_>>
    UnitreeLinkTest::low_state_publisher_;
std::unique_ptr<unitree::robot::ChannelPublisher<Clock_>>
    UnitreeLinkTest::clock_publisher_;
std::unique_ptr<unitree::robot::ChannelPublisher<PoseStamped_>>
    UnitreeLinkTest::pelvis_pose_publisher_;
std::unique_ptr<unitree::robot::ChannelSubscriber<LowCmd_>>
    UnitreeLinkTest::low_cmd_subscriber_;
Mailbox<g1_bridge::RobotState> UnitreeLinkTest::robot_states_;
Mailbox<g1_bridge::TimeStamp> UnitreeLinkTest::clocks_;
Mailbox<g1_bridge::StampedPose> UnitreeLinkTest::pelvis_poses_;
Mailbox<LowCmd_> UnitreeLinkTest::low_cmds_;

// These fail before the SDK is touched, so they need no fixture.
TEST(UnitreeLinkArguments, RejectsUnusableInterfaceAndDomain) {
  EXPECT_THROW(g1_bridge::UnitreeLink(TEST_DOMAIN_ID, ""),
               std::invalid_argument);
  EXPECT_THROW(g1_bridge::UnitreeLink(TEST_DOMAIN_ID, "g1_no_such_if"),
               std::runtime_error);
  EXPECT_THROW(g1_bridge::UnitreeLink(-1, TEST_INTERFACE),
               std::invalid_argument);
  EXPECT_THROW(g1_bridge::UnitreeLink(233, TEST_INTERFACE),
               std::invalid_argument);
}

TEST_F(UnitreeLinkTest, ForwardsLowState) {
  LowState_ msg;
  msg.mode_pr() = 1;
  msg.mode_machine() = 5;
  msg.tick() = 123456;
  msg.imu_state().quaternion() = {0.5F, -0.5F, 0.5F, -0.5F};
  msg.imu_state().gyroscope() = {0.1F, 0.2F, 0.3F};
  msg.imu_state().accelerometer() = {0.4F, 0.5F, 9.8F};
  for (std::size_t i = 0; i < g1_bridge::kNumMotorSlots; ++i) {
    const auto n = static_cast<float>(i + 1);
    msg.motor_state()[i].q() = 0.01F * n;
    msg.motor_state()[i].dq() = -0.02F * n;
    msg.motor_state()[i].tau_est() = 0.3F * n;
  }

  robot_states_.Clear();
  const auto state = robot_states_.WaitFor(
      [&msg] { low_state_publisher_->Write(msg); },
      [](const g1_bridge::RobotState &received) {
        return received.tick == 123456;
      });

  ASSERT_TRUE(state.has_value());
  EXPECT_EQ(state->mode_pr, 1);
  EXPECT_EQ(state->mode_machine, 5);
  EXPECT_EQ(state->quaternion_wxyz, msg.imu_state().quaternion());
  EXPECT_EQ(state->gyroscope, msg.imu_state().gyroscope());
  EXPECT_EQ(state->accelerometer, msg.imu_state().accelerometer());
  for (std::size_t i = 0; i < g1_bridge::kNumMotorSlots; ++i) {
    const auto n = static_cast<float>(i + 1);
    EXPECT_FLOAT_EQ(state->q[i], 0.01F * n) << i;
    EXPECT_FLOAT_EQ(state->dq[i], -0.02F * n) << i;
    EXPECT_FLOAT_EQ(state->tau_est[i], 0.3F * n) << i;
  }
}

TEST_F(UnitreeLinkTest, ForwardsSimulatorClock) {
  Clock_ msg;
  msg.clock().sec() = 42;
  msg.clock().nanosec() = 123456789;

  clocks_.Clear();
  const auto stamp = clocks_.WaitFor(
      [&msg] { clock_publisher_->Write(msg); },
      [](const g1_bridge::TimeStamp &received) { return received.sec == 42; });

  ASSERT_TRUE(stamp.has_value());
  EXPECT_EQ(stamp->nanosec, 123456789U);
}

TEST_F(UnitreeLinkTest, ForwardsSimulatorPelvisPose) {
  PoseStamped_ msg;
  msg.header().stamp().sec() = 7;
  msg.header().stamp().nanosec() = 250000000;
  msg.header().frame_id() = "world";
  msg.pose().position().x() = 1.5;
  msg.pose().position().y() = -2.5;
  msg.pose().position().z() = 0.79;
  msg.pose().orientation().x() = 0.1;
  msg.pose().orientation().y() = 0.2;
  msg.pose().orientation().z() = 0.3;
  msg.pose().orientation().w() = 0.9;

  pelvis_poses_.Clear();
  const auto pose = pelvis_poses_.WaitFor(
      [&msg] { pelvis_pose_publisher_->Write(msg); },
      [](const g1_bridge::StampedPose &received) {
        return received.stamp.sec == 7;
      });

  ASSERT_TRUE(pose.has_value());
  EXPECT_EQ(pose->stamp.nanosec, 250000000U);
  EXPECT_EQ(pose->frame_id, "world");
  EXPECT_DOUBLE_EQ(pose->position[0], 1.5);
  EXPECT_DOUBLE_EQ(pose->position[1], -2.5);
  EXPECT_DOUBLE_EQ(pose->position[2], 0.79);
  EXPECT_DOUBLE_EQ(pose->orientation_xyzw[0], 0.1);
  EXPECT_DOUBLE_EQ(pose->orientation_xyzw[1], 0.2);
  EXPECT_DOUBLE_EQ(pose->orientation_xyzw[2], 0.3);
  EXPECT_DOUBLE_EQ(pose->orientation_xyzw[3], 0.9);
}

// A link that only listens, like sim_relay_node's, has no rt/lowcmd writer.
TEST_F(UnitreeLinkTest, RefusesLowCommandUntilEnabled) {
  EXPECT_FALSE(wrote_before_enable_);
}

TEST_F(UnitreeLinkTest, WritesLowCommandWithItsCrc) {
  g1_bridge::LowCommandInputs inputs;
  inputs.state = g1_bridge::ControlState::kControl;
  inputs.mode_pr = 1;
  inputs.mode_machine = 5;
  for (std::size_t i = 0; i < g1_bridge::kNumMotors; ++i) {
    const auto n = static_cast<float>(i + 1);
    inputs.target_positions[i] = 0.01F * n;
    inputs.target_velocities[i] = -0.02F * n;
    inputs.target_efforts[i] = 0.3F * n;
  }
  std::array<g1_bridge::PidGains, g1_bridge::kNumMotors> gains;
  for (std::size_t i = 0; i < g1_bridge::kNumMotors; ++i) {
    gains[i] = {40.0 + static_cast<double>(i), 2.0 + static_cast<double>(i)};
  }
  const auto command = g1_bridge::BuildLowCommand(inputs, gains);
  ASSERT_NE(command.crc, 0U);

  low_cmds_.Clear();
  const auto msg = low_cmds_.WaitFor(
      [&command] { EXPECT_TRUE(link_->WriteLowCommand(command)); },
      [](const LowCmd_ &received) { return received.mode_machine() == 5; });

  ASSERT_TRUE(msg.has_value());
  EXPECT_EQ(msg->mode_pr(), 1);
  EXPECT_EQ(msg->crc(), command.crc);
  EXPECT_EQ(msg->crc(), g1_bridge::ComputeLowCommandCrc(command));
  EXPECT_EQ(msg->reserve(), command.reserve);
  for (std::size_t i = 0; i < g1_bridge::kNumMotorSlots; ++i) {
    const auto &sent = command.motor_cmd[i];
    const auto &received = msg->motor_cmd()[i];
    EXPECT_EQ(received.mode(), sent.mode) << i;
    EXPECT_EQ(received.q(), sent.q) << i;
    EXPECT_EQ(received.dq(), sent.dq) << i;
    EXPECT_EQ(received.tau(), sent.tau) << i;
    EXPECT_EQ(received.kp(), sent.kp) << i;
    EXPECT_EQ(received.kd(), sent.kd) << i;
    EXPECT_EQ(received.reserve(), sent.reserve) << i;
  }
  // The enabled joints carry what the builder was given.
  EXPECT_EQ(msg->motor_cmd()[0].mode(), 1);
  EXPECT_FLOAT_EQ(msg->motor_cmd()[28].q(), 0.29F);
  EXPECT_FLOAT_EQ(msg->motor_cmd()[28].kp(), 68.0F);
  EXPECT_EQ(msg->motor_cmd()[29].mode(), 0);
}

}  // namespace
