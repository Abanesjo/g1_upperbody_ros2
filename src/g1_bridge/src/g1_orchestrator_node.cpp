#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/string.hpp"

namespace {

constexpr std::size_t G1_NUM_MOTOR = 29;
constexpr int JOY_DAMP_BUTTON = 4;
constexpr int JOY_CONTROL_TOGGLE_BUTTON = 5;
constexpr double NEUTRAL_DURATION_SECONDS = 3.0;

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

constexpr std::array<float, G1_NUM_MOTOR> DEFAULT_JOINT_POS = {
    -0.1F, 0.0F, 0.0F, 0.3F, -0.2F, 0.0F,
    -0.1F, 0.0F, 0.0F, 0.3F, -0.2F, 0.0F,
    0.0F, 0.0F, 0.0F,
    0.37F, 0.62F, 0.0F, 0.82F, 0.0F, 0.0F, 0.0F,
    0.33F, -0.67F, 0.0F, 1.01F, 0.0F, 0.0F, 0.0F,
};

enum class OrchestratorState {
  kNeutral,
  kControl,
  kDamp,
};

}  // namespace

class G1OrchestratorNode : public rclcpp::Node {
 public:
  G1OrchestratorNode() : Node("g1_orchestrator_node") {
    for (std::size_t i = 0; i < JOINT_NAMES.size(); ++i) {
      joint_map_[JOINT_NAMES[i]] = i;
    }

    this->declare_parameter<double>("publish_rate", 100.0);
    const double rate =
        this->get_parameter("publish_rate").as_double();

    this->declare_parameter<std::string>("initial_mode", "neutral");
    const std::string initial_mode =
        this->get_parameter("initial_mode").as_string();
    if (initial_mode == "control") {
      state_ = OrchestratorState::kControl;
    } else if (initial_mode != "neutral") {
      RCLCPP_WARN(this->get_logger(),
                  "Unknown initial_mode '%s', defaulting to neutral",
                  initial_mode.c_str());
    }

    auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(1))
                          .best_effort()
                          .durability_volatile();

    safe_cmd_pub_ =
        this->create_publisher<sensor_msgs::msg::JointState>(
            "/joint_commands_safe", sensor_qos);

    state_pub_ =
        this->create_publisher<std_msgs::msg::String>(
            "/orchestrator/state",
            rclcpp::QoS(1).transient_local());

    joint_cmd_sub_ =
        this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_commands", sensor_qos,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
              JointCommandCallback(msg);
            });
    joint_states_sub_ =
        this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", sensor_qos,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
              JointStatesCallback(msg);
            });
    joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", sensor_qos,
        [this](const sensor_msgs::msg::Joy::SharedPtr msg) {
          JoyCallback(msg);
        });

    publish_timer_ = rclcpp::create_timer(
        this, this->get_clock(),
        std::chrono::microseconds(static_cast<int64_t>(1e6 / rate)),
        [this] { PublishOrchestratedCommand(); });

    PublishState();

    RCLCPP_INFO(this->get_logger(), "G1 orchestrator node started");
  }

 private:
  static bool IsButtonPressed(const sensor_msgs::msg::Joy &msg,
                              std::size_t index) {
    return index < msg.buttons.size() && msg.buttons[index] != 0;
  }

  static double Clamp(double value, double low, double high) {
    if (value < low) {
      return low;
    }
    if (value > high) {
      return high;
    }
    return value;
  }

  void SetState(OrchestratorState new_state) {
    if (new_state == state_) {
      return;
    }
    state_ = new_state;
    PublishState();
  }

  void PublishState() {
    std_msgs::msg::String msg;
    switch (state_) {
      case OrchestratorState::kNeutral:
        msg.data = "neutral";
        break;
      case OrchestratorState::kControl:
        msg.data = "control";
        break;
      case OrchestratorState::kDamp:
        msg.data = "damp";
        break;
    }
    state_pub_->publish(msg);
  }

  void StartNeutralRamp(const rclcpp::Time &now) {
    neutral_start_positions_ = current_positions_;
    neutral_start_time_ = now;
    neutral_ramp_active_ = true;
    SetState(OrchestratorState::kNeutral);
  }

  void JoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg) {
    const bool damp_pressed = IsButtonPressed(*msg, JOY_DAMP_BUTTON);
    if (damp_pressed && state_ != OrchestratorState::kDamp) {
      neutral_ramp_active_ = false;
      SetState(OrchestratorState::kDamp);
      RCLCPP_ERROR(this->get_logger(),
                   "Joy button[%d] pressed - entering latched damp state",
                   JOY_DAMP_BUTTON);
      return;
    }

    if (state_ == OrchestratorState::kDamp) {
      return;
    }

    const bool control_pressed =
        IsButtonPressed(*msg, JOY_CONTROL_TOGGLE_BUTTON);
    const bool control_rising_edge =
        control_pressed && !last_control_button_pressed_;
    last_control_button_pressed_ = control_pressed;

    if (!control_rising_edge) {
      return;
    }

    if (state_ == OrchestratorState::kControl) {
      if (has_state_) {
        StartNeutralRamp(this->get_clock()->now());
      } else {
        SetState(OrchestratorState::kNeutral);
      }
      RCLCPP_INFO(this->get_logger(), "Switching to neutral state");
      return;
    }

    if (neutral_ramp_active_) {
      RCLCPP_WARN(this->get_logger(),
                  "Ignoring control toggle while neutral ramp is active");
      return;
    }

    if (!has_latest_control_cmd_) {
      RCLCPP_WARN(this->get_logger(),
                  "Ignoring control toggle before any /joint_commands message");
      return;
    }

    SetState(OrchestratorState::kControl);
    RCLCPP_INFO(this->get_logger(), "Switching to control state");
  }

  void JointStatesCallback(
      const sensor_msgs::msg::JointState::SharedPtr msg) {
    const bool had_state = has_state_;
    has_state_ = true;
    for (std::size_t i = 0; i < msg->name.size(); ++i) {
      const auto it = joint_map_.find(msg->name[i]);
      if (it == joint_map_.end()) {
        continue;
      }
      if (i < msg->position.size()) {
        current_positions_[it->second] =
            static_cast<float>(msg->position[i]);
      }
    }
    if (!had_state) {
      target_positions_ = current_positions_;
      target_velocities_.fill(0.0F);
      target_efforts_.fill(0.0F);
      if (state_ == OrchestratorState::kControl) {
        RCLCPP_INFO(this->get_logger(),
                    "Received first /joint_states - starting in control mode");
      } else if (state_ == OrchestratorState::kDamp) {
        RCLCPP_INFO(this->get_logger(),
                    "Received first /joint_states - starting in damp mode");
      } else {
        StartNeutralRamp(this->get_clock()->now());
        RCLCPP_INFO(this->get_logger(),
                    "Received first /joint_states - starting neutral ramp");
      }
    }
  }

  void JointCommandCallback(
      const sensor_msgs::msg::JointState::SharedPtr msg) {
    if (!has_state_) {
      RCLCPP_WARN(this->get_logger(),
                  "Received /joint_commands before any /joint_states - ignoring");
      return;
    }

    if (!has_latest_control_cmd_) {
      target_positions_ = current_positions_;
    }
    target_velocities_.fill(0.0F);
    target_efforts_.fill(0.0F);

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
      if (has_position && i < msg->position.size()) {
        target_positions_[it->second] = static_cast<float>(msg->position[i]);
      }
      if (has_velocity && i < msg->velocity.size()) {
        target_velocities_[it->second] = static_cast<float>(msg->velocity[i]);
      }
      if (has_effort && i < msg->effort.size()) {
        target_efforts_[it->second] = static_cast<float>(msg->effort[i]);
      }
    }

    has_latest_control_cmd_ = true;
  }

  void BuildNeutralCommand(sensor_msgs::msg::JointState &cmd,
                           const rclcpp::Time &now) {
    double ratio = 1.0;
    if (neutral_ramp_active_) {
      ratio = Clamp(
          (now - neutral_start_time_).seconds() / NEUTRAL_DURATION_SECONDS,
          0.0, 1.0);
      if (ratio >= 1.0) {
        neutral_ramp_active_ = false;
      }
    }
    for (std::size_t i = 0; i < JOINT_NAMES.size(); ++i) {
      cmd.name.push_back(JOINT_NAMES[i]);
      cmd.position.push_back(
          neutral_ramp_active_
              ? static_cast<double>(
                    (1.0 - ratio) * neutral_start_positions_[i] +
                    ratio * DEFAULT_JOINT_POS[i])
              : static_cast<double>(DEFAULT_JOINT_POS[i]));
      cmd.velocity.push_back(0.0);
      cmd.effort.push_back(0.0);
    }
  }

  void BuildDampCommand(sensor_msgs::msg::JointState &cmd) {
    for (std::size_t i = 0; i < JOINT_NAMES.size(); ++i) {
      cmd.name.push_back(JOINT_NAMES[i]);
      cmd.position.push_back(static_cast<double>(current_positions_[i]));
      cmd.velocity.push_back(0.0);
      cmd.effort.push_back(0.0);
    }
  }

  void BuildControlCommand(sensor_msgs::msg::JointState &cmd) {
    for (std::size_t i = 0; i < JOINT_NAMES.size(); ++i) {
      cmd.name.push_back(JOINT_NAMES[i]);
      cmd.position.push_back(static_cast<double>(target_positions_[i]));
      cmd.velocity.push_back(static_cast<double>(target_velocities_[i]));
      cmd.effort.push_back(static_cast<double>(target_efforts_[i]));
    }
  }

  void PublishOrchestratedCommand() {
    if (!has_state_) {
      return;
    }

    sensor_msgs::msg::JointState cmd;
    cmd.header.stamp = this->get_clock()->now();
    cmd.name.reserve(G1_NUM_MOTOR);
    cmd.position.reserve(G1_NUM_MOTOR);
    cmd.velocity.reserve(G1_NUM_MOTOR);
    cmd.effort.reserve(G1_NUM_MOTOR);

    if (state_ == OrchestratorState::kDamp) {
      BuildDampCommand(cmd);
    } else if (state_ == OrchestratorState::kControl) {
      BuildControlCommand(cmd);
    } else {
      BuildNeutralCommand(cmd, this->get_clock()->now());
    }

    safe_cmd_pub_->publish(cmd);
  }

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr safe_cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
      joint_states_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;

  std::unordered_map<std::string, std::size_t> joint_map_;
  std::array<float, G1_NUM_MOTOR> current_positions_{};
  std::array<float, G1_NUM_MOTOR> neutral_start_positions_{};
  std::array<float, G1_NUM_MOTOR> target_positions_{};
  std::array<float, G1_NUM_MOTOR> target_velocities_{};
  std::array<float, G1_NUM_MOTOR> target_efforts_{};
  rclcpp::Time neutral_start_time_{};
  OrchestratorState state_{OrchestratorState::kNeutral};
  bool has_state_{false};
  bool has_latest_control_cmd_{false};
  bool neutral_ramp_active_{false};
  bool last_control_button_pressed_{false};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  int ret = 0;
  try {
    rclcpp::spin(std::make_shared<G1OrchestratorNode>());
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("g1_orchestrator_node"), "%s", e.what());
    ret = 1;
  }
  rclcpp::shutdown();
  return ret;
}
