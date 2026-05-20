#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "common/motor_crc_hg.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"

namespace {

constexpr std::size_t G1_NUM_MOTOR = 29;

const std::array<std::string, G1_NUM_MOTOR> JOINT_NAMES = {
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

}  // namespace

class G1BridgeNode : public rclcpp::Node {
 public:
  G1BridgeNode() : Node("g1_bridge_node") {
    for (std::size_t i = 0; i < JOINT_NAMES.size(); ++i) {
      const auto &name = JOINT_NAMES[i];
      joint_map_[name] = i;

      const auto kp_param = "gains." + name + ".kp";
      const auto kd_param = "gains." + name + ".kd";
      this->declare_parameter<double>(kp_param, 100.0);
      this->declare_parameter<double>(kd_param, 3.0);
      gains_[i] = {
          this->get_parameter(kp_param).as_double(),
          this->get_parameter(kd_param).as_double(),
      };
    }

    auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(1))
                          .best_effort()
                          .durability_volatile();

    joint_states_pub_ =
        this->create_publisher<sensor_msgs::msg::JointState>("/joint_states",
                                                             sensor_qos);
    imu_pub_ =
        this->create_publisher<sensor_msgs::msg::Imu>("/imu", sensor_qos);
    lowcmd_pub_ =
        this->create_publisher<unitree_hg::msg::LowCmd>("/lowcmd",
                                                        sensor_qos);

    lowstate_sub_ = this->create_subscription<unitree_hg::msg::LowState>(
        "/lowstate", sensor_qos,
        [this](const unitree_hg::msg::LowState::SharedPtr msg) {
          LowStateCallback(msg);
        });
    joint_cmd_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_commands", sensor_qos,
        [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
          JointCommandCallback(msg);
        });

    republish_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(2), [this] { RepublishLowCmd(); });

    RCLCPP_INFO(this->get_logger(), "G1 bridge node started");
  }

 private:
  void LowStateCallback(const unitree_hg::msg::LowState::SharedPtr msg) {
    mode_machine_ = msg->mode_machine;
    mode_pr_ = msg->mode_pr;
    has_state_ = true;

    const auto stamp = this->get_clock()->now();

    sensor_msgs::msg::JointState joint_state;
    joint_state.header.stamp = stamp;
    joint_state.name.reserve(G1_NUM_MOTOR);
    joint_state.position.reserve(G1_NUM_MOTOR);
    joint_state.velocity.reserve(G1_NUM_MOTOR);
    joint_state.effort.reserve(G1_NUM_MOTOR);

    for (std::size_t i = 0; i < JOINT_NAMES.size(); ++i) {
      const auto &motor = msg->motor_state[i];
      joint_state.name.push_back(JOINT_NAMES[i]);
      joint_state.position.push_back(static_cast<double>(motor.q));
      joint_state.velocity.push_back(static_cast<double>(motor.dq));
      joint_state.effort.push_back(static_cast<double>(motor.tau_est));
    }
    joint_states_pub_->publish(joint_state);

    sensor_msgs::msg::Imu imu;
    imu.header.stamp = stamp;
    imu.header.frame_id = "torso_link";
    imu.orientation.w = msg->imu_state.quaternion[0];
    imu.orientation.x = msg->imu_state.quaternion[1];
    imu.orientation.y = msg->imu_state.quaternion[2];
    imu.orientation.z = msg->imu_state.quaternion[3];
    imu.angular_velocity.x = msg->imu_state.gyroscope[0];
    imu.angular_velocity.y = msg->imu_state.gyroscope[1];
    imu.angular_velocity.z = msg->imu_state.gyroscope[2];
    imu.linear_acceleration.x = msg->imu_state.accelerometer[0];
    imu.linear_acceleration.y = msg->imu_state.accelerometer[1];
    imu.linear_acceleration.z = msg->imu_state.accelerometer[2];
    imu_pub_->publish(imu);
  }

  void JointCommandCallback(
      const sensor_msgs::msg::JointState::SharedPtr msg) {
    if (!has_state_) {
      RCLCPP_WARN(this->get_logger(),
                  "Received /joint_commands before any /lowstate - ignoring");
      return;
    }

    unitree_hg::msg::LowCmd cmd;
    cmd.mode_pr = mode_pr_;
    cmd.mode_machine = mode_machine_;

    for (std::size_t i = 0; i < JOINT_NAMES.size(); ++i) {
      auto &motor_cmd = cmd.motor_cmd[i];
      motor_cmd.mode = 1;
      motor_cmd.kp = static_cast<float>(gains_[i].first);
      motor_cmd.kd = static_cast<float>(gains_[i].second);
    }

    const bool has_position = !msg->position.empty();
    const bool has_velocity = !msg->velocity.empty();
    const bool has_effort = !msg->effort.empty();

    for (std::size_t i = 0; i < msg->name.size(); ++i) {
      const auto it = joint_map_.find(msg->name[i]);
      if (it == joint_map_.end()) {
        RCLCPP_WARN(this->get_logger(), "Unknown joint name in command: %s",
                    msg->name[i].c_str());
        continue;
      }

      auto &motor_cmd = cmd.motor_cmd[it->second];
      if (has_position && i < msg->position.size()) {
        motor_cmd.q = static_cast<float>(msg->position[i]);
      }
      if (has_velocity && i < msg->velocity.size()) {
        motor_cmd.dq = static_cast<float>(msg->velocity[i]);
      }
      if (has_effort && i < msg->effort.size()) {
        motor_cmd.tau = static_cast<float>(msg->effort[i]);
      }
    }

    get_crc(cmd);
    latest_lowcmd_ = cmd;
    has_latest_lowcmd_ = true;
    lowcmd_pub_->publish(cmd);
  }

  void RepublishLowCmd() {
    if (has_latest_lowcmd_) {
      lowcmd_pub_->publish(latest_lowcmd_);
    }
  }

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<unitree_hg::msg::LowCmd>::SharedPtr lowcmd_pub_;
  rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr lowstate_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_sub_;
  rclcpp::TimerBase::SharedPtr republish_timer_;

  std::unordered_map<std::string, std::size_t> joint_map_;
  std::array<std::pair<double, double>, G1_NUM_MOTOR> gains_{};
  unitree_hg::msg::LowCmd latest_lowcmd_;
  uint8_t mode_machine_{};
  uint8_t mode_pr_{};
  bool has_state_{false};
  bool has_latest_lowcmd_{false};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<G1BridgeNode>());
  rclcpp::shutdown();
  return 0;
}
