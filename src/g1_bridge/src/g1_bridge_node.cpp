#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "common/motor_crc_hg.h"
#include "nlohmann/json.hpp"
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "unitree_api/msg/request.hpp"
#include "unitree_api/msg/response.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"

namespace {

constexpr std::size_t G1_NUM_MOTOR = 29;
constexpr int32_t UT_ROBOT_SUCCESS = 0;
constexpr int32_t UT_ROBOT_TASK_TIMEOUT = -1;
constexpr int32_t UT_ROBOT_TASK_UNKNOWN_ERROR = -2;
constexpr int64_t MOTION_SWITCHER_API_ID_CHECK_MODE = 1001;
constexpr int64_t MOTION_SWITCHER_API_ID_RELEASE_MODE = 1003;
constexpr int MOTION_RELEASE_MAX_ATTEMPTS = 3;
constexpr int JOY_DAMP_BUTTON = 4;
constexpr int JOY_CONTROL_TOGGLE_BUTTON = 5;
constexpr double NEUTRAL_DURATION_SECONDS = 3.0;
constexpr uint8_t MODE_PR = 0;
constexpr float NEUTRAL_LEG_KP = 100.0F;
constexpr float NEUTRAL_UPPER_KP = 50.0F;
constexpr float NEUTRAL_KD = 1.0F;
constexpr float DAMP_KP = 0.0F;
constexpr float DAMP_KD = 2.0F;
constexpr auto MOTION_CALL_TIMEOUT = std::chrono::seconds(5);
constexpr auto MOTION_RELEASE_RETRY_DELAY = std::chrono::seconds(2);
constexpr char MOTION_SWITCH_REQUEST_TOPIC[] = "/api/motion_switcher/request";
constexpr char MOTION_SWITCH_RESPONSE_TOPIC[] = "/api/motion_switcher/response";

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

int64_t GetMonotonicNanoseconds() {
  timespec ts {};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

std::string QueryServiceName(const std::string &form, const std::string &name) {
  if (form == "0") {
    if (name == "normal") {
      return "sport_mode";
    }
    if (name == "ai") {
      return "ai_sport";
    }
    if (name == "advanced") {
      return "advanced_sport";
    }
  } else {
    if (name == "ai-w") {
      return "wheeled_sport(go2W)";
    }
    if (name == "normal-w") {
      return "wheeled_sport(b2W)";
    }
  }
  return name;
}

enum class OrchestratorState {
  kNeutral,
  kControl,
  kDamp,
};

}  // namespace

class G1BridgeNode : public rclcpp::Node {
 public:
  G1BridgeNode() : Node("g1_bridge_node") {
    const bool simulator = this->declare_parameter<bool>("simulator", false);
    if (simulator) {
      RCLCPP_WARN(this->get_logger(),
                  "Simulator mode enabled - skipping Unitree G1 motion "
                  "service deactivation");
    } else if (!DeactivateMotionService()) {
      throw std::runtime_error(
          "failed to deactivate Unitree G1 motion service");
    }

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
    joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", sensor_qos,
        [this](const sensor_msgs::msg::Joy::SharedPtr msg) {
          JoyCallback(msg);
        });

    republish_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(2), [this] { PublishOrchestratedLowCmd(); });

    RCLCPP_INFO(this->get_logger(), "G1 bridge node started");
  }

 private:
  int32_t CallMotionSwitcher(int64_t api_id, nlohmann::json *response_json) {
    using Request = unitree_api::msg::Request;
    using Response = unitree_api::msg::Response;

    if (!motion_switch_request_pub_) {
      motion_switch_request_pub_ =
          this->create_publisher<Request>(MOTION_SWITCH_REQUEST_TOPIC,
                                          rclcpp::QoS(1));
    }

    Request req;
    req.header.identity.id = GetMonotonicNanoseconds();
    req.header.identity.api_id = api_id;

    const auto identity_id = req.header.identity.id;
    auto response_promise = std::make_shared<std::promise<Response>>();
    auto response_set = std::make_shared<std::atomic_bool>(false);
    auto response_future = response_promise->get_future();

    auto response_sub = this->create_subscription<Response>(
        MOTION_SWITCH_RESPONSE_TOPIC, rclcpp::QoS(1),
        [identity_id, response_promise, response_set](
            const Response::SharedPtr msg) {
          if (msg->header.identity.id != identity_id) {
            return;
          }
          if (!response_set->exchange(true)) {
            response_promise->set_value(*msg);
          }
        });

    motion_switch_request_pub_->publish(req);

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(this->get_node_base_interface());
    const auto status =
        executor.spin_until_future_complete(response_future,
                                            MOTION_CALL_TIMEOUT);
    executor.remove_node(this->get_node_base_interface());

    if (status == rclcpp::FutureReturnCode::TIMEOUT) {
      return UT_ROBOT_TASK_TIMEOUT;
    }
    if (status != rclcpp::FutureReturnCode::SUCCESS) {
      return UT_ROBOT_TASK_UNKNOWN_ERROR;
    }

    const auto response = response_future.get();
    if (response.header.status.code != UT_ROBOT_SUCCESS) {
      return response.header.status.code;
    }

    if (response_json != nullptr) {
      try {
        *response_json = nlohmann::json::parse(response.data);
      } catch (const nlohmann::json::exception &e) {
        RCLCPP_ERROR(this->get_logger(),
                     "Failed to parse motion switch response JSON: %s",
                     e.what());
        return UT_ROBOT_TASK_UNKNOWN_ERROR;
      }
    }

    return UT_ROBOT_SUCCESS;
  }

  int32_t CheckMotionMode(std::string &form, std::string &name) {
    nlohmann::json response;
    const auto ret =
        CallMotionSwitcher(MOTION_SWITCHER_API_ID_CHECK_MODE, &response);
    if (ret != UT_ROBOT_SUCCESS) {
      return ret;
    }

    try {
      response.at("form").get_to(form);
      response.at("name").get_to(name);
    } catch (const nlohmann::json::exception &e) {
      RCLCPP_ERROR(this->get_logger(),
                   "Motion switch CheckMode response is missing form/name: %s",
                   e.what());
      return UT_ROBOT_TASK_UNKNOWN_ERROR;
    }

    return UT_ROBOT_SUCCESS;
  }

  int32_t ReleaseMotionMode() {
    return CallMotionSwitcher(MOTION_SWITCHER_API_ID_RELEASE_MODE, nullptr);
  }

  bool DeactivateMotionService() {
    RCLCPP_INFO(this->get_logger(),
                "Checking Unitree G1 motion service before enabling lowcmd");

    std::string form;
    std::string name;
    auto ret = CheckMotionMode(form, name);
    if (ret != UT_ROBOT_SUCCESS) {
      RCLCPP_FATAL(this->get_logger(),
                   "CheckMode failed before lowcmd startup. Error code: %d",
                   ret);
      return false;
    }

    for (int attempt = 1; !name.empty() &&
                          attempt <= MOTION_RELEASE_MAX_ATTEMPTS;
         ++attempt) {
      const auto service_name = QueryServiceName(form, name);
      RCLCPP_WARN(this->get_logger(),
                  "Unitree motion service '%s' is active; release attempt "
                  "%d/%d",
                  service_name.c_str(), attempt, MOTION_RELEASE_MAX_ATTEMPTS);

      ret = ReleaseMotionMode();
      if (ret != UT_ROBOT_SUCCESS) {
        RCLCPP_FATAL(this->get_logger(),
                     "ReleaseMode failed before lowcmd startup. Error code: %d",
                     ret);
        return false;
      }

      std::this_thread::sleep_for(MOTION_RELEASE_RETRY_DELAY);

      ret = CheckMotionMode(form, name);
      if (ret != UT_ROBOT_SUCCESS) {
        RCLCPP_FATAL(this->get_logger(),
                     "CheckMode failed after ReleaseMode. Error code: %d",
                     ret);
        return false;
      }
    }

    if (!name.empty()) {
      const auto service_name = QueryServiceName(form, name);
      RCLCPP_FATAL(this->get_logger(),
                   "Unitree motion service '%s' is still active after %d "
                   "release attempts; refusing to publish lowcmd",
                   service_name.c_str(), MOTION_RELEASE_MAX_ATTEMPTS);
      return false;
    }

    RCLCPP_INFO(this->get_logger(),
                "Unitree motion service is deactivated; enabling lowcmd");
    return true;
  }

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

  void StartNeutralRamp(const rclcpp::Time &now) {
    neutral_start_positions_ = current_positions_;
    neutral_start_time_ = now;
    neutral_ramp_active_ = true;
    state_ = OrchestratorState::kNeutral;
  }

  void JoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg) {
    const bool damp_pressed = IsButtonPressed(*msg, JOY_DAMP_BUTTON);
    if (damp_pressed && state_ != OrchestratorState::kDamp) {
      state_ = OrchestratorState::kDamp;
      neutral_ramp_active_ = false;
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
        state_ = OrchestratorState::kNeutral;
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

    state_ = OrchestratorState::kControl;
    RCLCPP_INFO(this->get_logger(), "Switching to control state");
  }

  void BuildNeutralCommand(unitree_hg::msg::LowCmd &cmd,
                           const rclcpp::Time &now) {
    double ratio = 1.0;
    if (neutral_ramp_active_) {
      ratio =
          Clamp((now - neutral_start_time_).seconds() / NEUTRAL_DURATION_SECONDS,
                0.0, 1.0);
      if (ratio >= 1.0) {
        neutral_ramp_active_ = false;
      }
    }

    cmd.mode_pr = MODE_PR;
    cmd.mode_machine = mode_machine_;
    for (std::size_t i = 0; i < JOINT_NAMES.size(); ++i) {
      auto &motor_cmd = cmd.motor_cmd[i];
      motor_cmd.mode = 1;
      motor_cmd.tau = 0.0F;
      motor_cmd.q =
          neutral_ramp_active_
              ? static_cast<float>((1.0 - ratio) * neutral_start_positions_[i])
              : 0.0F;
      motor_cmd.dq = 0.0F;
      motor_cmd.kp = (i < 13) ? NEUTRAL_LEG_KP : NEUTRAL_UPPER_KP;
      motor_cmd.kd = NEUTRAL_KD;
    }
  }

  void BuildDampCommand(unitree_hg::msg::LowCmd &cmd) {
    cmd.mode_pr = MODE_PR;
    cmd.mode_machine = mode_machine_;
    for (std::size_t i = 0; i < JOINT_NAMES.size(); ++i) {
      auto &motor_cmd = cmd.motor_cmd[i];
      motor_cmd.mode = 1;
      motor_cmd.tau = 0.0F;
      motor_cmd.q = current_positions_[i];
      motor_cmd.dq = 0.0F;
      motor_cmd.kp = DAMP_KP;
      motor_cmd.kd = DAMP_KD;
    }
  }

  void LowStateCallback(const unitree_hg::msg::LowState::SharedPtr msg) {
    const bool had_state = has_state_;
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
      current_positions_[i] = static_cast<float>(motor.q);
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

    if (!had_state) {
      StartNeutralRamp(stamp);
      RCLCPP_INFO(this->get_logger(),
                  "Received first /lowstate - starting neutral ramp");
    }
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

    latest_control_lowcmd_ = cmd;
    has_latest_control_cmd_ = true;
  }

  void PublishOrchestratedLowCmd() {
    if (!has_state_) {
      return;
    }

    unitree_hg::msg::LowCmd cmd;
    if (state_ == OrchestratorState::kDamp) {
      BuildDampCommand(cmd);
    } else if (state_ == OrchestratorState::kControl &&
               has_latest_control_cmd_) {
      cmd = latest_control_lowcmd_;
    } else {
      BuildNeutralCommand(cmd, this->get_clock()->now());
    }

    get_crc(cmd);
    lowcmd_pub_->publish(cmd);
  }

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<unitree_hg::msg::LowCmd>::SharedPtr lowcmd_pub_;
  rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr lowstate_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::TimerBase::SharedPtr republish_timer_;
  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr
      motion_switch_request_pub_;

  std::unordered_map<std::string, std::size_t> joint_map_;
  std::array<std::pair<double, double>, G1_NUM_MOTOR> gains_{};
  std::array<float, G1_NUM_MOTOR> current_positions_{};
  std::array<float, G1_NUM_MOTOR> neutral_start_positions_{};
  unitree_hg::msg::LowCmd latest_control_lowcmd_;
  rclcpp::Time neutral_start_time_{};
  uint8_t mode_machine_{};
  uint8_t mode_pr_{};
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
    rclcpp::spin(std::make_shared<G1BridgeNode>());
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("g1_bridge_node"), "%s", e.what());
    ret = 1;
  }
  rclcpp::shutdown();
  return ret;
}
