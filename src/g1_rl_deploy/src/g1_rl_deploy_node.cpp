#include <array>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "onnxruntime_cxx_api.h"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "unitree_hg/msg/low_state.hpp"

static constexpr int NUM_MOTOR = 29;
static constexpr int NUM_UPPER_BODY = 8;
static constexpr int OBS_DIM = 106;

// Joint names in motor index order (0-28)
static const std::array<std::string, NUM_MOTOR> JOINT_NAMES = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
};

static const std::array<std::string, NUM_UPPER_BODY> UPPER_BODY_NAMES = {
    "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint",
};

static constexpr std::array<int, NUM_UPPER_BODY> UPPER_BODY_INDICES = {
    13, 14, 15, 16, 18, 22, 23, 25
};

// ---------------------------------------------------------------------------
// ONNX Policy wrapper
// ---------------------------------------------------------------------------
class OnnxPolicy {
public:
    OnnxPolicy(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "g1_policy") {
        session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            auto type_info = session_->GetInputTypeInfo(i);
            input_shapes_.push_back(type_info.GetTensorTypeAndShapeInfo().GetShape());
            auto name = session_->GetInputNameAllocated(i, allocator_);
            input_name_strs_.push_back(name.get());
            size_t size = 1;
            for (auto dim : input_shapes_.back()) size *= dim;
            input_sizes_.push_back(size);
        }
        for (auto& s : input_name_strs_) input_names_.push_back(s.c_str());
        auto out_type = session_->GetOutputTypeInfo(0);
        output_shape_ = out_type.GetTensorTypeAndShapeInfo().GetShape();
        auto out_name = session_->GetOutputNameAllocated(0, allocator_);
        output_name_str_ = out_name.get();
        output_name_ = output_name_str_.c_str();
    }

    std::vector<float> infer(std::vector<float>& obs) {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::vector<Ort::Value> input_tensors;
        for (size_t i = 0; i < input_names_.size(); ++i) {
            auto tensor = Ort::Value::CreateTensor<float>(
                memory_info, obs.data(), input_sizes_[i],
                input_shapes_[i].data(), input_shapes_[i].size());
            input_tensors.push_back(std::move(tensor));
        }
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), input_tensors.data(), input_tensors.size(),
            &output_name_, 1);
        auto* floatarr = output_tensors.front().GetTensorMutableData<float>();
        return std::vector<float>(floatarr, floatarr + output_shape_[1]);
    }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<std::string> input_name_strs_;
    std::vector<const char*> input_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<int64_t> input_sizes_;
    std::string output_name_str_;
    const char* output_name_;
    std::vector<int64_t> output_shape_;
};

// ---------------------------------------------------------------------------
// ROS2 Node
// ---------------------------------------------------------------------------
class G1RLDeployNode : public rclcpp::Node {
public:
    G1RLDeployNode() : Node("g1_rl_deploy_node"), time_(0.0), global_phase_(0.0f),
                       running_policy_(false), state_received_(false) {

        // Declare parameters
        this->declare_parameter<std::string>("model_path",
            "/workspace/ros2_ws/install/g1_rl_deploy/share/g1_rl_deploy/models/policy.onnx");
        this->declare_parameter<double>("control_dt", 0.02);
        this->declare_parameter<double>("standup_duration", 3.0);
        this->declare_parameter<double>("gait_period", 0.6);
        this->declare_parameter<std::vector<double>>("cmd_vel_limits.lin_vel_x", {-0.5, 1.0});
        this->declare_parameter<std::vector<double>>("cmd_vel_limits.lin_vel_y", {-0.5, 0.5});
        this->declare_parameter<std::vector<double>>("cmd_vel_limits.ang_vel_z", {-1.0, 1.0});
        this->declare_parameter<std::vector<double>>("default_joint_pos", std::vector<double>{
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
             0.0, 0.0, 0.0, 0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
             0.35,-0.18, 0.0, 0.87, 0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>("action_scale", std::vector<double>{
            0.55, 0.35, 0.55, 0.35, 0.44, 0.44, 0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
            0.35, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,
            0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07});

        // Read parameters
        std::string model_path = this->get_parameter("model_path").as_string();
        control_dt_ = this->get_parameter("control_dt").as_double();
        standup_duration_ = this->get_parameter("standup_duration").as_double();
        gait_period_ = this->get_parameter("gait_period").as_double();
        default_pos_ = this->get_parameter("default_joint_pos").as_double_array();
        action_scale_ = this->get_parameter("action_scale").as_double_array();

        auto lim_x = this->get_parameter("cmd_vel_limits.lin_vel_x").as_double_array();
        auto lim_y = this->get_parameter("cmd_vel_limits.lin_vel_y").as_double_array();
        auto lim_z = this->get_parameter("cmd_vel_limits.ang_vel_z").as_double_array();
        vel_limit_x_ = {static_cast<float>(lim_x[0]), static_cast<float>(lim_x[1])};
        vel_limit_y_ = {static_cast<float>(lim_y[0]), static_cast<float>(lim_y[1])};
        vel_limit_z_ = {static_cast<float>(lim_z[0]), static_cast<float>(lim_z[1])};

        // Init upper body defaults
        for (int i = 0; i < NUM_UPPER_BODY; ++i)
            upper_body_cmd_[i] = static_cast<float>(default_pos_[UPPER_BODY_INDICES[i]]);
        for (int i = 0; i < NUM_UPPER_BODY; ++i)
            upper_body_name_map_[UPPER_BODY_NAMES[i]] = i;

        // Load policy
        RCLCPP_INFO(this->get_logger(), "Loading policy: %s", model_path.c_str());
        policy_ = std::make_unique<OnnxPolicy>(model_path);
        last_action_.resize(NUM_MOTOR, 0.0f);

        auto sensor_qos = rclcpp::SensorDataQoS().keep_last(1);

        // Subscribers
        joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", sensor_qos,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) { JointStatesCallback(msg); });

        // IMU comes from /lowstate (not available in JointState)
        lowstate_sub_ = this->create_subscription<unitree_hg::msg::LowState>(
            "/lowstate", sensor_qos,
            [this](const unitree_hg::msg::LowState::SharedPtr msg) { LowStateCallback(msg); });

        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", sensor_qos,
            [this](const geometry_msgs::msg::Twist::SharedPtr msg) { CmdVelCallback(msg); });

        upper_body_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/upper_body_targets", sensor_qos,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) { UpperBodyCallback(msg); });

        // Publisher: JointState with 29 joint target positions → bridge
        joint_cmd_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/joint_commands", 10);

        // 50 Hz control timer
        control_timer_ = this->create_wall_timer(
            std::chrono::microseconds(static_cast<int>(control_dt_ * 1e6)),
            [this] { Control(); });

        RCLCPP_INFO(this->get_logger(), "Waiting for /joint_states, /lowstate, /cmd_vel, /upper_body_targets...");
    }

private:
    void CmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        vel_cmd_[0] = std::clamp(static_cast<float>(msg->linear.x), vel_limit_x_[0], vel_limit_x_[1]);
        vel_cmd_[1] = std::clamp(static_cast<float>(msg->linear.y), vel_limit_y_[0], vel_limit_y_[1]);
        vel_cmd_[2] = std::clamp(static_cast<float>(msg->angular.z), vel_limit_z_[0], vel_limit_z_[1]);
    }

    void UpperBodyCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        for (size_t j = 0; j < msg->name.size() && j < msg->position.size(); ++j) {
            auto it = upper_body_name_map_.find(msg->name[j]);
            if (it != upper_body_name_map_.end())
                upper_body_cmd_[it->second] = static_cast<float>(msg->position[j]);
        }
    }

    void JointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        for (size_t j = 0; j < msg->name.size(); ++j) {
            // Find motor index for this joint name
            for (int i = 0; i < NUM_MOTOR; ++i) {
                if (msg->name[j] == JOINT_NAMES[i]) {
                    if (j < msg->position.size()) motor_q_[i] = msg->position[j];
                    if (j < msg->velocity.size()) motor_dq_[i] = msg->velocity[j];
                    break;
                }
            }
        }
        state_received_ = true;
    }

    void LowStateCallback(const unitree_hg::msg::LowState::SharedPtr msg) {
        // Only used for IMU (not available in JointState)
        for (int i = 0; i < 4; ++i)
            imu_quat_[i] = msg->imu_state.quaternion[i];
        for (int i = 0; i < 3; ++i)
            imu_gyro_[i] = msg->imu_state.gyroscope[i];
    }

    void Control() {
        if (!state_received_) return;

        sensor_msgs::msg::JointState cmd;
        cmd.header.stamp = this->now();
        cmd.name.assign(JOINT_NAMES.begin(), JOINT_NAMES.end());
        cmd.position.resize(NUM_MOTOR, 0.0);

        time_ += control_dt_;

        if (time_ < standup_duration_) {
            float ratio = std::clamp(static_cast<float>(time_ / standup_duration_), 0.0f, 1.0f);
            for (int i = 0; i < NUM_MOTOR; ++i)
                cmd.position[i] = (1.0 - ratio) * motor_q_[i] + ratio * default_pos_[i];

            if (!running_policy_) {
                static bool printed = false;
                if (!printed) {
                    RCLCPP_INFO(this->get_logger(), "Phase 1: Standing up (%.0fs)...", standup_duration_);
                    printed = true;
                }
            }
        } else {
            if (!running_policy_) {
                running_policy_ = true;
                RCLCPP_INFO(this->get_logger(), "Phase 2: Policy active");
            }

            std::vector<float> obs;
            obs.reserve(OBS_DIM);

            for (int i = 0; i < 3; ++i) obs.push_back(imu_gyro_[i]);

            Eigen::Quaternionf q(imu_quat_[0], imu_quat_[1], imu_quat_[2], imu_quat_[3]);
            Eigen::Vector3f gravity_b = q.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
            obs.push_back(gravity_b.x()); obs.push_back(gravity_b.y()); obs.push_back(gravity_b.z());

            for (int i = 0; i < 3; ++i) obs.push_back(vel_cmd_[i]);

            global_phase_ += static_cast<float>(control_dt_ / gait_period_);
            global_phase_ = std::fmod(global_phase_, 1.0f);
            float cmd_norm = std::sqrt(vel_cmd_[0]*vel_cmd_[0] + vel_cmd_[1]*vel_cmd_[1] + vel_cmd_[2]*vel_cmd_[2]);
            if (cmd_norm < 0.1f) { obs.push_back(0.0f); obs.push_back(0.0f); }
            else { obs.push_back(std::sin(global_phase_ * 2.0f * M_PI)); obs.push_back(std::cos(global_phase_ * 2.0f * M_PI)); }

            for (int i = 0; i < NUM_MOTOR; ++i) obs.push_back(motor_q_[i] - static_cast<float>(default_pos_[i]));
            for (int i = 0; i < NUM_MOTOR; ++i) obs.push_back(motor_dq_[i]);
            for (int i = 0; i < NUM_MOTOR; ++i) obs.push_back(last_action_[i]);
            for (int i = 0; i < NUM_UPPER_BODY; ++i) obs.push_back(upper_body_cmd_[i]);

            auto raw_action = policy_->infer(obs);
            last_action_ = raw_action;

            for (int i = 0; i < NUM_MOTOR; ++i)
                cmd.position[i] = raw_action[i] * action_scale_[i] + default_pos_[i];

            static int print_counter = 0;
            if (++print_counter % 250 == 0) {
                RCLCPP_INFO(this->get_logger(), "t=%.1f cmd_vel=[%.2f,%.2f,%.2f]",
                            time_, vel_cmd_[0], vel_cmd_[1], vel_cmd_[2]);
            }
        }

        joint_cmd_pub_->publish(cmd);
    }

    // ROS2
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
    rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr lowstate_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr upper_body_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    // Policy
    std::unique_ptr<OnnxPolicy> policy_;
    std::vector<float> last_action_;

    // Parameters
    std::vector<double> default_pos_, action_scale_;
    double control_dt_, standup_duration_, gait_period_;
    std::array<float, 2> vel_limit_x_, vel_limit_y_, vel_limit_z_;

    // Upper body
    std::array<float, NUM_UPPER_BODY> upper_body_cmd_ = {};
    std::unordered_map<std::string, int> upper_body_name_map_;

    // State
    double time_;
    float global_phase_;
    bool running_policy_;
    bool state_received_;
    std::array<float, NUM_MOTOR> motor_q_ = {};
    std::array<float, NUM_MOTOR> motor_dq_ = {};
    std::array<float, 4> imu_quat_ = {};
    std::array<float, 3> imu_gyro_ = {};
    std::array<float, 3> vel_cmd_ = {0.0f, 0.0f, 0.0f};
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<G1RLDeployNode>());
    rclcpp::shutdown();
    return 0;
}
