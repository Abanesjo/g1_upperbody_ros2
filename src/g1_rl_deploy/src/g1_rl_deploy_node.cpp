#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "onnxruntime_cxx_api.h"

#include "rclcpp/rclcpp.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"
#include "geometry_msgs/msg/twist.hpp"

#include "g1_rl_deploy/motor_crc_hg.h"

// ---------------------------------------------------------------------------
// Constants from deploy.yaml (Unitree-G1-Flat velocity policy)
// ---------------------------------------------------------------------------
static constexpr int NUM_MOTOR = 29;
static constexpr int OBS_DIM = 98;
static constexpr float STEP_DT = 0.02f;        // 50 Hz policy
static constexpr float STANDUP_DURATION = 3.0f;
static constexpr float GAIT_PERIOD = 0.6f;

static constexpr std::array<float, NUM_MOTOR> KP = {
    40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f,
    40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f,
    40.2f, 28.5f, 28.5f,
    14.3f, 14.3f, 14.3f, 14.3f, 14.3f, 16.8f, 16.8f,
    14.3f, 14.3f, 14.3f, 14.3f, 14.3f, 16.8f, 16.8f
};

static constexpr std::array<float, NUM_MOTOR> KD = {
    2.6f, 6.3f, 2.6f, 6.3f, 1.8f, 1.8f,
    2.6f, 6.3f, 2.6f, 6.3f, 1.8f, 1.8f,
    2.6f, 1.8f, 1.8f,
    0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 1.1f, 1.1f,
    0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 1.1f, 1.1f
};

static constexpr std::array<float, NUM_MOTOR> DEFAULT_POS = {
    -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
    -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
     0.0f, 0.0f, 0.0f,
     0.35f, 0.18f, 0.0f, 0.87f, 0.0f, 0.0f, 0.0f,
     0.35f,-0.18f, 0.0f, 0.87f, 0.0f, 0.0f, 0.0f
};

static constexpr std::array<float, NUM_MOTOR> ACTION_SCALE = {
    0.55f, 0.35f, 0.55f, 0.35f, 0.44f, 0.44f,
    0.55f, 0.35f, 0.55f, 0.35f, 0.44f, 0.44f,
    0.35f, 0.44f, 0.44f,
    0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.07f, 0.07f,
    0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.07f, 0.07f
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
        for (auto& s : input_name_strs_)
            input_names_.push_back(s.c_str());

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
                       running_policy_(false), mode_machine_(0), state_received_(false) {

        this->declare_parameter<std::string>("model_path",
            "/workspace/ros2_ws/install/g1_rl_deploy/share/g1_rl_deploy/models/policy.onnx");

        std::string model_path = this->get_parameter("model_path").as_string();
        RCLCPP_INFO(this->get_logger(), "Loading policy: %s", model_path.c_str());
        policy_ = std::make_unique<OnnxPolicy>(model_path);
        last_action_.resize(NUM_MOTOR, 0.0f);

        // Subscribers
        lowstate_sub_ = this->create_subscription<unitree_hg::msg::LowState>(
            "lowstate", 10,
            [this](const unitree_hg::msg::LowState::SharedPtr msg) { LowStateCallback(msg); });

        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            [this](const geometry_msgs::msg::Twist::SharedPtr msg) { CmdVelCallback(msg); });

        // Publisher
        lowcmd_pub_ = this->create_publisher<unitree_hg::msg::LowCmd>("/lowcmd", 10);

        // 50 Hz: policy inference, updates latest_cmd_
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            [this] { Control(); });

        // 500 Hz: republish latest command for stable PD control
        publish_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(2),
            [this] { PublishCmd(); });

        RCLCPP_INFO(this->get_logger(), "Waiting for /lowstate and /cmd_vel...");
    }

private:
    void CmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        vel_cmd_[0] = std::clamp(static_cast<float>(msg->linear.x), -0.5f, 1.0f);
        vel_cmd_[1] = std::clamp(static_cast<float>(msg->linear.y), -0.5f, 0.5f);
        vel_cmd_[2] = std::clamp(static_cast<float>(msg->angular.z), -1.0f, 1.0f);
    }

    void LowStateCallback(const unitree_hg::msg::LowState::SharedPtr msg) {
        for (int i = 0; i < NUM_MOTOR; ++i) {
            motor_q_[i] = msg->motor_state[i].q;
            motor_dq_[i] = msg->motor_state[i].dq;
        }
        for (int i = 0; i < 4; ++i)
            imu_quat_[i] = msg->imu_state.quaternion[i];
        for (int i = 0; i < 3; ++i)
            imu_gyro_[i] = msg->imu_state.gyroscope[i];

        mode_machine_ = msg->mode_machine;
        state_received_ = true;
    }

    void Control() {
        if (!state_received_) return;

        unitree_hg::msg::LowCmd cmd;
        cmd.mode_pr = 0;
        cmd.mode_machine = mode_machine_;

        for (int i = 0; i < NUM_MOTOR; ++i) {
            cmd.motor_cmd[i].mode = 1;
            cmd.motor_cmd[i].kp = KP[i];
            cmd.motor_cmd[i].kd = KD[i];
            cmd.motor_cmd[i].tau = 0.0f;
            cmd.motor_cmd[i].dq = 0.0f;
        }

        time_ += STEP_DT;

        if (time_ < STANDUP_DURATION) {
            float ratio = std::clamp(static_cast<float>(time_ / STANDUP_DURATION), 0.0f, 1.0f);
            for (int i = 0; i < NUM_MOTOR; ++i)
                cmd.motor_cmd[i].q = (1.0f - ratio) * motor_q_[i] + ratio * DEFAULT_POS[i];

            if (!running_policy_) {
                static bool printed = false;
                if (!printed) {
                    RCLCPP_INFO(this->get_logger(), "Phase 1: Standing up (%.0fs)...", STANDUP_DURATION);
                    printed = true;
                }
            }
        } else {
            if (!running_policy_) {
                running_policy_ = true;
                RCLCPP_INFO(this->get_logger(), "Phase 2: Policy active");
            }

            // Build observation (98 dims)
            std::vector<float> obs;
            obs.reserve(OBS_DIM);

            // 1. base_ang_vel (3)
            for (int i = 0; i < 3; ++i)
                obs.push_back(imu_gyro_[i]);

            // 2. projected_gravity (3)
            Eigen::Quaternionf q(imu_quat_[0], imu_quat_[1], imu_quat_[2], imu_quat_[3]);
            Eigen::Vector3f gravity_b = q.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
            obs.push_back(gravity_b.x());
            obs.push_back(gravity_b.y());
            obs.push_back(gravity_b.z());

            // 3. velocity_commands (3)
            for (int i = 0; i < 3; ++i)
                obs.push_back(vel_cmd_[i]);

            // 4. gait_phase (2)
            global_phase_ += STEP_DT / GAIT_PERIOD;
            global_phase_ = std::fmod(global_phase_, 1.0f);
            float cmd_norm = std::sqrt(vel_cmd_[0]*vel_cmd_[0] + vel_cmd_[1]*vel_cmd_[1] + vel_cmd_[2]*vel_cmd_[2]);
            if (cmd_norm < 0.1f) {
                obs.push_back(0.0f);
                obs.push_back(0.0f);
            } else {
                obs.push_back(std::sin(global_phase_ * 2.0f * M_PI));
                obs.push_back(std::cos(global_phase_ * 2.0f * M_PI));
            }

            // 5. joint_pos_rel (29)
            for (int i = 0; i < NUM_MOTOR; ++i)
                obs.push_back(motor_q_[i] - DEFAULT_POS[i]);

            // 6. joint_vel_rel (29)
            for (int i = 0; i < NUM_MOTOR; ++i)
                obs.push_back(motor_dq_[i]);

            // 7. last_action (29)
            for (int i = 0; i < NUM_MOTOR; ++i)
                obs.push_back(last_action_[i]);

            // Infer
            auto raw_action = policy_->infer(obs);
            last_action_ = raw_action;

            // target_q = raw * scale + offset
            for (int i = 0; i < NUM_MOTOR; ++i)
                cmd.motor_cmd[i].q = raw_action[i] * ACTION_SCALE[i] + DEFAULT_POS[i];

            static int print_counter = 0;
            if (++print_counter % 250 == 0) {
                RCLCPP_INFO(this->get_logger(), "t=%.1f cmd_vel=[%.2f,%.2f,%.2f]",
                            time_, vel_cmd_[0], vel_cmd_[1], vel_cmd_[2]);
            }
        }

        get_crc(cmd);
        latest_cmd_ = cmd;
        cmd_ready_ = true;
    }

    void PublishCmd() {
        if (!cmd_ready_) return;
        lowcmd_pub_->publish(latest_cmd_);
    }

    // ROS2
    rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr lowstate_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Publisher<unitree_hg::msg::LowCmd>::SharedPtr lowcmd_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::TimerBase::SharedPtr publish_timer_;

    // Policy
    std::unique_ptr<OnnxPolicy> policy_;
    std::vector<float> last_action_;

    // State
    double time_;
    float global_phase_;
    bool running_policy_;
    bool state_received_;
    bool cmd_ready_ = false;
    uint8_t mode_machine_;
    std::array<float, NUM_MOTOR> motor_q_ = {};
    std::array<float, NUM_MOTOR> motor_dq_ = {};
    std::array<float, 4> imu_quat_ = {};
    std::array<float, 3> imu_gyro_ = {};
    std::array<float, 3> vel_cmd_ = {0.0f, 0.0f, 0.0f};
    unitree_hg::msg::LowCmd latest_cmd_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<G1RLDeployNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
