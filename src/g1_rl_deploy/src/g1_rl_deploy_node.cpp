#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "onnxruntime_cxx_api.h"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/twist.hpp"

static constexpr int NUM_MOTOR = 29;
static constexpr int NUM_ACTION = 29;  // all joints (policy controls full body)
static constexpr int NUM_UPPER_BODY_CMD = 8;  // upper body command targets
static constexpr int OBS_DIM = 106;  // 3+3+3+2+29+29+29+8

// All 29 joint names in motor index order
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

// Upper body controlled joint names (CBF-filtered targets fed as observation)
static constexpr int UPPER_BODY_INDICES[NUM_UPPER_BODY_CMD] = {
    13, 14, 15, 16, 18, 22, 23, 25  // waist_roll, waist_pitch, L/R shoulder pitch/roll, L/R elbow
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
// ROS2 Node — Full-Body RL Policy Deploy (velocity + upper body tracking)
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
        // Action scale for all 29 joints
        this->declare_parameter<std::vector<double>>("action_scale", std::vector<double>(NUM_ACTION, 0.44));

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

        // Load policy
        RCLCPP_INFO(this->get_logger(), "Loading full-body policy: %s", model_path.c_str());
        policy_ = std::make_unique<OnnxPolicy>(model_path);
        last_action_.resize(NUM_ACTION, 0.0f);

        auto qos = rclcpp::SensorDataQoS().keep_last(1);

        // Subscribers
        joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", qos,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) { JointStatesCallback(msg); });

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu", qos,
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) { ImuCallback(msg); });

        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", qos,
            [this](const geometry_msgs::msg::Twist::SharedPtr msg) { CmdVelCallback(msg); });

        upper_body_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/upper_body_targets", qos,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) { UpperBodyCallback(msg); });

        // Publisher: all 29 joint commands
        joint_cmd_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/joint_commands", qos);

        // 50 Hz control timer
        control_timer_ = this->create_wall_timer(
            std::chrono::microseconds(static_cast<int>(control_dt_ * 1e6)),
            [this] { Control(); });

        // Initialize upper body command targets to defaults
        for (int i = 0; i < NUM_UPPER_BODY_CMD; ++i)
            upper_body_cmd_[i] = static_cast<float>(default_pos_[UPPER_BODY_INDICES[i]]);

        RCLCPP_INFO(this->get_logger(), "Waiting for /joint_states, /imu, /cmd_vel, /upper_body_targets...");
    }

private:
    void CmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        vel_cmd_[0] = std::clamp(static_cast<float>(msg->linear.x), vel_limit_x_[0], vel_limit_x_[1]);
        vel_cmd_[1] = std::clamp(static_cast<float>(msg->linear.y), vel_limit_y_[0], vel_limit_y_[1]);
        vel_cmd_[2] = std::clamp(static_cast<float>(msg->angular.z), vel_limit_z_[0], vel_limit_z_[1]);
    }

    void JointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        for (size_t j = 0; j < msg->name.size(); ++j) {
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

    void ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        imu_quat_[0] = msg->orientation.w;
        imu_quat_[1] = msg->orientation.x;
        imu_quat_[2] = msg->orientation.y;
        imu_quat_[3] = msg->orientation.z;
        imu_gyro_[0] = msg->angular_velocity.x;
        imu_gyro_[1] = msg->angular_velocity.y;
        imu_gyro_[2] = msg->angular_velocity.z;
    }

    void UpperBodyCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        for (size_t j = 0; j < msg->name.size(); ++j) {
            for (int i = 0; i < NUM_UPPER_BODY_CMD; ++i) {
                if (msg->name[j] == JOINT_NAMES[UPPER_BODY_INDICES[i]]) {
                    if (j < msg->position.size())
                        upper_body_cmd_[i] = static_cast<float>(msg->position[j]);
                    break;
                }
            }
        }
    }

    void Control() {
        if (!state_received_) return;

        sensor_msgs::msg::JointState cmd;
        cmd.header.stamp = this->now();

        time_ += control_dt_;

        if (time_ < standup_duration_) {
            // Standup: publish all 29 joints interpolating to default
            cmd.name.assign(JOINT_NAMES.begin(), JOINT_NAMES.end());
            cmd.position.resize(NUM_MOTOR, 0.0);
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
                RCLCPP_INFO(this->get_logger(), "Phase 2: Full-body policy active");
            }

            // Build observation (106 dims)
            std::vector<float> obs;
            obs.reserve(OBS_DIM);

            // 1. base_ang_vel (3)
            for (int i = 0; i < 3; ++i) obs.push_back(imu_gyro_[i]);

            // 2. projected_gravity (3)
            Eigen::Quaternionf q(imu_quat_[0], imu_quat_[1], imu_quat_[2], imu_quat_[3]);
            Eigen::Vector3f gravity_b = q.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
            obs.push_back(gravity_b.x()); obs.push_back(gravity_b.y()); obs.push_back(gravity_b.z());

            // 3. velocity_commands (3)
            for (int i = 0; i < 3; ++i) obs.push_back(vel_cmd_[i]);

            // 4. gait_phase (2) — sin/cos
            global_phase_ += static_cast<float>(control_dt_ / gait_period_);
            global_phase_ = std::fmod(global_phase_, 1.0f);
            float cmd_norm = std::sqrt(vel_cmd_[0]*vel_cmd_[0] + vel_cmd_[1]*vel_cmd_[1] + vel_cmd_[2]*vel_cmd_[2]);
            if (cmd_norm < 0.1f) { obs.push_back(0.0f); obs.push_back(0.0f); }
            else { obs.push_back(std::sin(global_phase_ * 2.0f * M_PI)); obs.push_back(std::cos(global_phase_ * 2.0f * M_PI)); }

            // 5. joint_pos_rel (29) — all joints
            for (int i = 0; i < NUM_MOTOR; ++i)
                obs.push_back(motor_q_[i] - static_cast<float>(default_pos_[i]));

            // 6. joint_vel (29) — all joints
            for (int i = 0; i < NUM_MOTOR; ++i)
                obs.push_back(motor_dq_[i]);

            // 7. last_action (29) — all joints
            for (int i = 0; i < NUM_ACTION; ++i)
                obs.push_back(last_action_[i]);

            // 8. upper_body_command (8) — CBF-filtered targets
            for (int i = 0; i < NUM_UPPER_BODY_CMD; ++i)
                obs.push_back(upper_body_cmd_[i]);

            // Infer — outputs 29 actions for all joints
            auto raw_action = policy_->infer(obs);
            last_action_ = raw_action;

            // Publish all 29 joints
            cmd.name.assign(JOINT_NAMES.begin(), JOINT_NAMES.end());
            cmd.position.resize(NUM_MOTOR);
            for (int i = 0; i < NUM_MOTOR; ++i)
                cmd.position[i] = raw_action[i] * action_scale_[i] + default_pos_[i];

            // Override 6 arm joints with CBF-filtered targets directly
            // Skip waist_roll (i=0) and waist_pitch (i=1) — policy controls those
            for (int i = 2; i < NUM_UPPER_BODY_CMD; ++i)
                cmd.position[UPPER_BODY_INDICES[i]] = upper_body_cmd_[i];

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
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
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
    std::array<float, NUM_UPPER_BODY_CMD> upper_body_cmd_ = {};
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<G1RLDeployNode>());
    rclcpp::shutdown();
    return 0;
}
