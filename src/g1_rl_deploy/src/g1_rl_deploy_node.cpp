#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "onnxruntime_cxx_api.h"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

static constexpr int NUM_MOTOR = 29;
static constexpr int NUM_LOWER_BODY_ACTION = 12;
static constexpr int NUM_POLICY_ACTION = NUM_LOWER_BODY_ACTION;
static constexpr int NUM_UPPER_BODY_CMD = 17;
static constexpr int OBS_DIM = 93;

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

// Training command order for Unitree-G1-LowerBody-Flat.
static constexpr int UPPER_BODY_INDICES[NUM_UPPER_BODY_CMD] = {
    12, 13, 14,
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28,
};

class OnnxPolicy {
public:
    explicit OnnxPolicy(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "g1_lower_body_policy") {
        session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

        if (session_->GetInputCount() != 1) {
            throw std::runtime_error("expected exactly one ONNX policy input");
        }
        if (session_->GetOutputCount() != 1) {
            throw std::runtime_error("expected exactly one ONNX policy output");
        }

        auto input_type = session_->GetInputTypeInfo(0);
        input_shape_ = input_type.GetTensorTypeAndShapeInfo().GetShape();
        auto input_name = session_->GetInputNameAllocated(0, allocator_);
        input_name_str_ = input_name.get();
        input_name_ = input_name_str_.c_str();
        input_size_ = TensorElementCount(input_shape_);

        auto output_type = session_->GetOutputTypeInfo(0);
        output_shape_ = output_type.GetTensorTypeAndShapeInfo().GetShape();
        auto output_name = session_->GetOutputNameAllocated(0, allocator_);
        output_name_str_ = output_name.get();
        output_name_ = output_name_str_.c_str();
        output_size_ = TensorElementCount(output_shape_);
    }

    int64_t input_size() const { return input_size_; }
    int64_t output_size() const { return output_size_; }

    std::vector<float> infer(const std::vector<float>& obs) {
        if (static_cast<int64_t>(obs.size()) != input_size_) {
            throw std::runtime_error("ONNX input size mismatch");
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(obs.data()), obs.size(),
            input_shape_.data(), input_shape_.size());

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            &input_name_, &input_tensor, 1,
            &output_name_, 1);

        auto* data = output_tensors.front().GetTensorMutableData<float>();
        return std::vector<float>(data, data + output_size_);
    }

private:
    static int64_t TensorElementCount(const std::vector<int64_t>& shape) {
        int64_t count = 1;
        for (const auto dim : shape) {
            if (dim <= 0) {
                throw std::runtime_error("dynamic ONNX tensor dimensions are not supported");
            }
            count *= dim;
        }
        return count;
    }

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_str_;
    const char* input_name_;
    std::vector<int64_t> input_shape_;
    int64_t input_size_;
    std::string output_name_str_;
    const char* output_name_;
    std::vector<int64_t> output_shape_;
    int64_t output_size_;
};

class G1RLDeployNode : public rclcpp::Node {
public:
    G1RLDeployNode()
        : Node("g1_rl_deploy_node"),
          time_(0.0),
          running_policy_(false),
          state_received_(false),
          imu_received_(false) {
        this->declare_parameter<std::string>(
            "model_path",
            "/workspace/ros2_ws/install/g1_rl_deploy/share/g1_rl_deploy/models/policy.onnx");
        this->declare_parameter<double>("control_dt", 0.02);
        this->declare_parameter<double>("standup_duration", 3.0);
        this->declare_parameter<double>("action_smoothing_alpha", 0.8);
        this->declare_parameter<std::vector<double>>("default_joint_pos", std::vector<double>{
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
             0.0,   0.0, 0.0,
             0.2,   0.6, 0.0, 0.6, 0.0, 0.0, 0.0,
             0.2,  -0.6, 0.0, 0.6, 0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>("action_scale", std::vector<double>{
            0.70, 0.45, 0.30, 0.90, 0.55, 0.25,
            0.70, 0.45, 0.30, 0.90, 0.55, 0.25});

        const auto model_path = this->get_parameter("model_path").as_string();
        control_dt_ = this->get_parameter("control_dt").as_double();
        standup_duration_ = this->get_parameter("standup_duration").as_double();
        action_smoothing_alpha_ = this->get_parameter("action_smoothing_alpha").as_double();
        default_pos_ = this->get_parameter("default_joint_pos").as_double_array();
        action_scale_ = this->get_parameter("action_scale").as_double_array();

        if (default_pos_.size() != NUM_MOTOR) {
            throw std::runtime_error("default_joint_pos must contain 29 values");
        }
        if (action_scale_.size() != NUM_LOWER_BODY_ACTION) {
            throw std::runtime_error("action_scale must contain 12 lower-body values");
        }
        if (action_smoothing_alpha_ < 0.0 || action_smoothing_alpha_ > 1.0) {
            throw std::runtime_error("action_smoothing_alpha must be in [0, 1]");
        }

        RCLCPP_INFO(this->get_logger(), "Loading lower-body policy: %s", model_path.c_str());
        policy_ = std::make_unique<OnnxPolicy>(model_path);
        if (policy_->input_size() != OBS_DIM) {
            throw std::runtime_error("policy input dimension is not 93");
        }
        if (policy_->output_size() != NUM_POLICY_ACTION) {
            throw std::runtime_error("policy output dimension is not 12");
        }

        last_lower_body_action_.fill(0.0f);
        smoothed_lower_body_target_.fill(0.0f);
        for (int i = 0; i < NUM_LOWER_BODY_ACTION; ++i) {
            smoothed_lower_body_target_[i] = static_cast<float>(default_pos_[i]);
        }
        for (int i = 0; i < NUM_UPPER_BODY_CMD; ++i) {
            upper_body_cmd_[i] = static_cast<float>(default_pos_[UPPER_BODY_INDICES[i]]);
        }

        auto qos = rclcpp::SensorDataQoS().keep_last(1);
        joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", qos,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
                JointStatesCallback(msg);
            });
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu", qos,
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                ImuCallback(msg);
            });
        upper_body_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/upper_body_targets", qos,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
                UpperBodyCallback(msg);
            });
        joint_cmd_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/joint_commands", qos);

        control_timer_ = this->create_wall_timer(
            std::chrono::microseconds(static_cast<int>(control_dt_ * 1e6)),
            [this] { Control(); });

        RCLCPP_INFO(
            this->get_logger(),
            "Waiting for /joint_states, /imu, and /upper_body_targets...");
    }

private:
    void JointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        for (size_t j = 0; j < msg->name.size(); ++j) {
            for (int i = 0; i < NUM_MOTOR; ++i) {
                if (msg->name[j] == JOINT_NAMES[i]) {
                    if (j < msg->position.size()) {
                        motor_q_[i] = static_cast<float>(msg->position[j]);
                    }
                    if (j < msg->velocity.size()) {
                        motor_dq_[i] = static_cast<float>(msg->velocity[j]);
                    }
                    break;
                }
            }
        }
        state_received_ = true;
    }

    void ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        imu_quat_[0] = static_cast<float>(msg->orientation.w);
        imu_quat_[1] = static_cast<float>(msg->orientation.x);
        imu_quat_[2] = static_cast<float>(msg->orientation.y);
        imu_quat_[3] = static_cast<float>(msg->orientation.z);
        imu_gyro_[0] = static_cast<float>(msg->angular_velocity.x);
        imu_gyro_[1] = static_cast<float>(msg->angular_velocity.y);
        imu_gyro_[2] = static_cast<float>(msg->angular_velocity.z);
        imu_received_ = true;
    }

    void UpperBodyCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        for (size_t j = 0; j < msg->name.size(); ++j) {
            for (int i = 1; i < NUM_UPPER_BODY_CMD; ++i) {
                if (msg->name[j] == JOINT_NAMES[UPPER_BODY_INDICES[i]]) {
                    if (j < msg->position.size()) {
                        upper_body_cmd_[i] = static_cast<float>(msg->position[j]);
                    }
                    break;
                }
            }
        }
        upper_body_cmd_[0] = static_cast<float>(default_pos_[12]);
    }

    std::vector<float> BuildObservation() {
        std::vector<float> obs;
        obs.reserve(OBS_DIM);

        for (int i = 0; i < 3; ++i) {
            obs.push_back(imu_gyro_[i]);
        }

        Eigen::Quaternionf q(imu_quat_[0], imu_quat_[1], imu_quat_[2], imu_quat_[3]);
        q.normalize();
        const Eigen::Vector3f gravity_b = q.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
        obs.push_back(gravity_b.x());
        obs.push_back(gravity_b.y());
        obs.push_back(gravity_b.z());

        for (int i = 0; i < NUM_MOTOR; ++i) {
            obs.push_back(motor_q_[i] - static_cast<float>(default_pos_[i]));
        }

        for (int i = 0; i < NUM_MOTOR; ++i) {
            obs.push_back(motor_dq_[i]);
        }

        for (const auto action : last_lower_body_action_) {
            obs.push_back(action);
        }

        for (const auto cmd : upper_body_cmd_) {
            obs.push_back(cmd);
        }

        if (obs.size() != OBS_DIM) {
            throw std::runtime_error("lower-body observation dimension mismatch");
        }
        return obs;
    }

    void Control() {
        if (!state_received_ || !imu_received_) {
            return;
        }

        sensor_msgs::msg::JointState cmd;
        cmd.header.stamp = this->now();
        cmd.name.assign(JOINT_NAMES.begin(), JOINT_NAMES.end());
        cmd.position.resize(NUM_MOTOR, 0.0);

        time_ += control_dt_;

        if (time_ < standup_duration_) {
            const float ratio = std::clamp(
                static_cast<float>(time_ / standup_duration_), 0.0f, 1.0f);
            for (int i = 0; i < NUM_MOTOR; ++i) {
                cmd.position[i] = (1.0f - ratio) * motor_q_[i] + ratio * default_pos_[i];
            }

            static bool printed = false;
            if (!printed) {
                RCLCPP_INFO(
                    this->get_logger(),
                    "Phase 1: standing to lower-body policy default pose (%.0fs)",
                    standup_duration_);
                printed = true;
            }
        } else {
            if (!running_policy_) {
                running_policy_ = true;
                RCLCPP_INFO(this->get_logger(), "Phase 2: lower-body policy active");
            }

            const auto obs = BuildObservation();
            const auto raw_action = policy_->infer(obs);

            for (int i = 0; i < NUM_LOWER_BODY_ACTION; ++i) {
                last_lower_body_action_[i] = raw_action[i];
                const float requested_target =
                    raw_action[i] * static_cast<float>(action_scale_[i]) +
                    static_cast<float>(default_pos_[i]);
                smoothed_lower_body_target_[i] =
                    static_cast<float>(action_smoothing_alpha_) * requested_target +
                    static_cast<float>(1.0 - action_smoothing_alpha_) *
                        smoothed_lower_body_target_[i];
            }

            for (int i = 0; i < NUM_MOTOR; ++i) {
                cmd.position[i] = default_pos_[i];
            }
            for (int i = 0; i < NUM_LOWER_BODY_ACTION; ++i) {
                cmd.position[i] = smoothed_lower_body_target_[i];
            }
            for (int i = 0; i < NUM_UPPER_BODY_CMD; ++i) {
                cmd.position[UPPER_BODY_INDICES[i]] = upper_body_cmd_[i];
            }
            cmd.position[12] = default_pos_[12];

            static int print_counter = 0;
            if (++print_counter % 250 == 0) {
                RCLCPP_INFO(
                    this->get_logger(),
                    "t=%.1f lower-body policy running", time_);
            }
        }

        joint_cmd_pub_->publish(cmd);
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr upper_body_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    std::unique_ptr<OnnxPolicy> policy_;

    std::vector<double> default_pos_;
    std::vector<double> action_scale_;
    double control_dt_;
    double standup_duration_;
    double action_smoothing_alpha_;

    double time_;
    bool running_policy_;
    bool state_received_;
    bool imu_received_;

    std::array<float, NUM_MOTOR> motor_q_ = {};
    std::array<float, NUM_MOTOR> motor_dq_ = {};
    std::array<float, 4> imu_quat_ = {1.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 3> imu_gyro_ = {};
    std::array<float, NUM_LOWER_BODY_ACTION> last_lower_body_action_ = {};
    std::array<float, NUM_LOWER_BODY_ACTION> smoothed_lower_body_target_ = {};
    std::array<float, NUM_UPPER_BODY_CMD> upper_body_cmd_ = {};
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<G1RLDeployNode>());
    rclcpp::shutdown();
    return 0;
}
