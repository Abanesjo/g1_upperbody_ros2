#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "g1_bridge/gravity_compensation.hpp"
#include "g1_bridge/low_cmd_builder.hpp"
#include "g1_bridge/unitree_link.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"

namespace {

using g1_bridge::ControlState;
using g1_bridge::kNumMotors;
using g1_bridge::kUpperBodyStart;

constexpr int32_t UT_ROBOT_SUCCESS = 0;
constexpr int MOTION_CHECK_MAX_ATTEMPTS = 3;
constexpr int MOTION_RELEASE_MAX_ATTEMPTS = 3;
constexpr auto MOTION_CHECK_RETRY_DELAY = std::chrono::seconds(1);
constexpr auto MOTION_RELEASE_RETRY_DELAY = std::chrono::seconds(2);
constexpr char LOOPBACK_INTERFACE[] = "lo";

const std::array<std::string, kNumMotors> JOINT_NAMES = {
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

}  // namespace

// Bridges the ROS graph to the robot. The robot side is unitree_sdk2 (DDS on
// one network interface), the ROS side is whatever RMW the graph uses. The SDK
// keeps delivering robot data on its own threads after ROS has shut down, so
// the SDK callbacks drop it from then on.
class G1BridgeNode : public rclcpp::Node {
 public:
  G1BridgeNode() : Node("g1_bridge_node") {
    const bool simulator = this->declare_parameter<bool>("simulator", false);
    this->declare_parameter<bool>("mujoco", false);
    wbc_ = this->declare_parameter<bool>("wbc", true);
    const bool gravity =
        this->declare_parameter<bool>("gravity", true);
    const auto urdf_path =
        this->declare_parameter<std::string>("urdf_path", "");
    const auto network_interface =
        this->declare_parameter<std::string>("network_interface",
                                             LOOPBACK_INTERFACE);
    const auto domain_id =
        static_cast<int>(this->declare_parameter<int>("domain_id", 0));
    gravity_enabled_ = g1_bridge::GravityFeedforwardEnabled(wbc_, gravity);

    if (gravity_enabled_) {
      gravity_compensation_ =
          std::make_unique<g1_bridge::GravityCompensation>(
              urdf_path,
              std::vector<std::string>(JOINT_NAMES.begin(),
                                       JOINT_NAMES.end()));
      measured_positions_.resize(kNumMotors);
      RCLCPP_INFO(this->get_logger(),
                  "Upper-body gravity compensation enabled in neutral and "
                  "control modes");
    } else if (wbc_) {
      RCLCPP_INFO(this->get_logger(),
                  "Gravity compensation inactive while wbc is enabled");
    } else {
      RCLCPP_INFO(this->get_logger(),
                  "Upper-body control uses classical PD without gravity "
                  "compensation");
    }

    RCLCPP_INFO(this->get_logger(),
                "Connecting to the robot with unitree_sdk2 on network "
                "interface '%s', DDS domain %d",
                network_interface.c_str(), domain_id);
    link_ = std::make_unique<g1_bridge::UnitreeLink>(domain_id,
                                                     network_interface);
    link_->EnableLowCommand();

    if (simulator) {
      RCLCPP_WARN(this->get_logger(),
                  "Simulator mode enabled - skipping Unitree G1 motion "
                  "service deactivation");
      if (network_interface != LOOPBACK_INTERFACE) {
        RCLCPP_WARN(this->get_logger(),
                    "Simulator mode on network interface '%s': lowcmd also "
                    "reaches a real robot on that network, whose motion "
                    "service is still active",
                    network_interface.c_str());
      }
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

    safe_cmd_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_commands_safe", sensor_qos,
        [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
          JointCommandSafeCallback(msg);
        });
    orchestrator_state_sub_ =
        this->create_subscription<std_msgs::msg::String>(
            "/orchestrator/state",
            rclcpp::QoS(1).transient_local(),
            [this](const std_msgs::msg::String::SharedPtr msg) {
              OrchestratorStateCallback(msg);
            });

    // The flag is shared, so the callback stays valid after this node is
    // gone. Set before the RMW shuts down, unlike rclcpp::ok().
    this->get_node_base_interface()->get_context()->add_pre_shutdown_callback(
        [shutting_down = ros_shutting_down_] { shutting_down->store(true); });

    link_->SubscribeRobotState(
        [this](const g1_bridge::RobotState &state) {
          RobotStateCallback(state);
        });

    republish_timer_ = rclcpp::create_timer(
        this, this->get_clock(),
        std::chrono::milliseconds(2), [this] { PublishLowCmd(); });

    RCLCPP_INFO(this->get_logger(), "G1 bridge node started");
  }

 private:
  // The first call can race DDS discovery of the service, so it is retried.
  bool CheckMotionMode(std::string &form, std::string &name,
                       const char *stage) {
    int32_t ret = UT_ROBOT_SUCCESS;
    for (int attempt = 1; attempt <= MOTION_CHECK_MAX_ATTEMPTS; ++attempt) {
      form.clear();
      name.clear();
      ret = link_->CheckMotionMode(form, name);
      if (ret == UT_ROBOT_SUCCESS) {
        return true;
      }
      if (attempt < MOTION_CHECK_MAX_ATTEMPTS) {
        RCLCPP_WARN(this->get_logger(),
                    "CheckMode attempt %d/%d failed with error code %d; "
                    "retrying",
                    attempt, MOTION_CHECK_MAX_ATTEMPTS, ret);
        std::this_thread::sleep_for(MOTION_CHECK_RETRY_DELAY);
      }
    }

    RCLCPP_FATAL(this->get_logger(),
                 "CheckMode failed %s: no usable answer from the Unitree "
                 "motion switcher. Error code: %d",
                 stage, ret);
    return false;
  }

  bool DeactivateMotionService() {
    RCLCPP_INFO(this->get_logger(),
                "Checking Unitree G1 motion service before enabling lowcmd");

    std::string form;
    std::string name;
    if (!CheckMotionMode(form, name, "before lowcmd startup")) {
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

      const auto ret = link_->ReleaseMotionMode();
      if (ret != UT_ROBOT_SUCCESS) {
        RCLCPP_FATAL(this->get_logger(),
                     "ReleaseMode failed before lowcmd startup. Error code: %d",
                     ret);
        return false;
      }

      std::this_thread::sleep_for(MOTION_RELEASE_RETRY_DELAY);

      if (!CheckMotionMode(form, name, "after ReleaseMode")) {
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

  // Runs on an SDK thread.
  void RobotStateCallback(const g1_bridge::RobotState &state) {
    if (ros_shutting_down_->load()) {
      return;
    }
    try {
      const auto stamp = this->get_clock()->now();

      sensor_msgs::msg::JointState joint_state;
      joint_state.header.stamp = stamp;
      joint_state.name.reserve(kNumMotors);
      joint_state.position.reserve(kNumMotors);
      joint_state.velocity.reserve(kNumMotors);
      joint_state.effort.reserve(kNumMotors);

      for (std::size_t i = 0; i < JOINT_NAMES.size(); ++i) {
        joint_state.name.push_back(JOINT_NAMES[i]);
        joint_state.position.push_back(static_cast<double>(state.q[i]));
        joint_state.velocity.push_back(static_cast<double>(state.dq[i]));
        joint_state.effort.push_back(static_cast<double>(state.tau_est[i]));
        if (gravity_enabled_) {
          measured_positions_[i] = static_cast<double>(state.q[i]);
        }
      }

      g1_bridge::JointArray gravity_torques{};
      if (gravity_enabled_) {
        try {
          const std::array<double, 4> base_orientation = {
              static_cast<double>(state.quaternion_wxyz[0]),
              static_cast<double>(state.quaternion_wxyz[1]),
              static_cast<double>(state.quaternion_wxyz[2]),
              static_cast<double>(state.quaternion_wxyz[3]),
          };
          const auto &gravity =
              gravity_compensation_->Compute(base_orientation,
                                             measured_positions_);
          for (std::size_t i = kUpperBodyStart; i < gravity.size(); ++i) {
            gravity_torques[i] = static_cast<float>(gravity[i]);
          }
        } catch (const std::exception &e) {
          gravity_torques.fill(0.0F);
          RCLCPP_WARN_THROTTLE(
              this->get_logger(), *this->get_clock(), 2000,
              "Failed to compute gravity compensation; using zero "
              "feedforward: %s",
              e.what());
        }
      }

      {
        std::lock_guard<std::mutex> lock(robot_state_mutex_);
        mode_machine_ = state.mode_machine;
        mode_pr_ = state.mode_pr;
        gravity_torques_ = gravity_torques;
        last_lowstate_time_ = std::chrono::steady_clock::now();
        has_state_ = true;
      }
      joint_states_pub_->publish(joint_state);

      sensor_msgs::msg::Imu imu;
      imu.header.stamp = stamp;
      imu.header.frame_id = "torso_link";
      imu.orientation.w = state.quaternion_wxyz[0];
      imu.orientation.x = state.quaternion_wxyz[1];
      imu.orientation.y = state.quaternion_wxyz[2];
      imu.orientation.z = state.quaternion_wxyz[3];
      imu.angular_velocity.x = state.gyroscope[0];
      imu.angular_velocity.y = state.gyroscope[1];
      imu.angular_velocity.z = state.gyroscope[2];
      imu.linear_acceleration.x = state.accelerometer[0];
      imu.linear_acceleration.y = state.accelerometer[1];
      imu.linear_acceleration.z = state.accelerometer[2];
      imu_pub_->publish(imu);
    } catch (const std::exception &e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "Failed to forward lowstate: %s", e.what());
    }
  }

  void JointCommandSafeCallback(
      const sensor_msgs::msg::JointState::SharedPtr msg) {
    {
      std::lock_guard<std::mutex> lock(robot_state_mutex_);
      if (!has_state_) {
        RCLCPP_WARN(this->get_logger(),
                    "Received /joint_commands_safe before any lowstate - "
                    "ignoring");
        return;
      }
    }

    target_velocities_.fill(0.0F);
    target_efforts_.fill(0.0F);

    const bool has_position = !msg->position.empty();
    const bool has_velocity = !msg->velocity.empty();
    const bool has_effort = !msg->effort.empty();

    for (std::size_t i = 0; i < msg->name.size(); ++i) {
      const auto it = joint_map_.find(msg->name[i]);
      if (it == joint_map_.end()) {
        RCLCPP_WARN(this->get_logger(), "Unknown joint name in safe command: %s",
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

    has_latest_safe_cmd_ = true;
  }

  void OrchestratorStateCallback(const std_msgs::msg::String::SharedPtr msg) {
    ControlState new_state = current_state_;
    if (msg->data == "neutral") {
      new_state = ControlState::kNeutral;
    } else if (msg->data == "control") {
      new_state = ControlState::kControl;
    } else if (msg->data == "damp") {
      new_state = ControlState::kDamp;
    } else {
      RCLCPP_WARN(this->get_logger(), "Unknown orchestrator state: %s",
                  msg->data.c_str());
      return;
    }
    if (new_state != current_state_) {
      current_state_ = new_state;
      RCLCPP_INFO(this->get_logger(), "Orchestrator state: %s",
                  msg->data.c_str());
    }
  }

  void PublishLowCmd() {
    if (!has_latest_safe_cmd_) {
      return;
    }

    g1_bridge::LowCommandInputs inputs;
    {
      std::lock_guard<std::mutex> lock(robot_state_mutex_);
      if (!has_state_) {
        return;
      }
      inputs.mode_pr = mode_pr_;
      inputs.mode_machine = mode_machine_;
      inputs.gravity_torques = gravity_torques_;
      inputs.lowstate_age =
          std::chrono::steady_clock::now() - last_lowstate_time_;
    }
    inputs.state = current_state_;
    inputs.target_positions = target_positions_;
    inputs.target_velocities = target_velocities_;
    inputs.target_efforts = target_efforts_;
    inputs.gravity_enabled = gravity_enabled_;

    if (gravity_enabled_ && current_state_ != ControlState::kDamp &&
        !g1_bridge::GravityFeedforwardApplies(inputs)) {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "LowState is stale; falling back to classical PD without gravity "
          "feedforward");
    }

    if (!link_->WriteLowCommand(
            g1_bridge::BuildLowCommand(inputs, gains_))) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "Failed to write lowcmd to the robot");
    }
  }

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr safe_cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr
      orchestrator_state_sub_;
  rclcpp::TimerBase::SharedPtr republish_timer_;
  std::shared_ptr<std::atomic_bool> ros_shutting_down_{
      std::make_shared<std::atomic_bool>(false)};

  std::unordered_map<std::string, std::size_t> joint_map_;
  std::array<g1_bridge::PidGains, kNumMotors> gains_{};
  g1_bridge::JointArray target_positions_{};
  g1_bridge::JointArray target_velocities_{};
  g1_bridge::JointArray target_efforts_{};
  ControlState current_state_{ControlState::kNeutral};
  bool has_latest_safe_cmd_{false};
  bool wbc_{true};
  bool gravity_enabled_{false};

  // Used by the SDK lowstate thread only.
  std::vector<double> measured_positions_;
  std::unique_ptr<g1_bridge::GravityCompensation> gravity_compensation_;

  // Written by the SDK lowstate thread, read by the ROS callbacks.
  std::mutex robot_state_mutex_;
  g1_bridge::JointArray gravity_torques_{};
  std::chrono::steady_clock::time_point last_lowstate_time_{};
  uint8_t mode_machine_{};
  uint8_t mode_pr_{};
  bool has_state_{false};

  // Declared last so that it is destroyed first: its destructor stops the SDK
  // threads that call back into the members above.
  std::unique_ptr<g1_bridge::UnitreeLink> link_;
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
