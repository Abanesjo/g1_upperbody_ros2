#include <array>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "onnxruntime_cxx_api.h"

#include "rclcpp/rclcpp.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

#include "g1_rl_deploy/motor_crc_hg.h"

static constexpr int NUM_MOTOR = 29;
static constexpr int NUM_UPPER_BODY = 8;
static constexpr int OBS_DIM = 106;  // 98 base + 8 upper_body_command

// Upper body joint names in observation order
static const std::array<std::string, NUM_UPPER_BODY> UPPER_BODY_NAMES = {
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
};

// Corresponding indices in the 29-DOF motor array
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

        // Declare and load parameters
        this->declare_parameter<std::string>("model_path",
            "/workspace/ros2_ws/install/g1_rl_deploy/share/g1_rl_deploy/models/policy.onnx");
        this->declare_parameter<double>("control_dt", 0.02);
        this->declare_parameter<double>("publish_dt", 0.002);
        this->declare_parameter<double>("standup_duration", 3.0);
        this->declare_parameter<double>("gait_period", 0.6);

        this->declare_parameter<std::vector<double>>("cmd_vel_limits.lin_vel_x",
            std::vector<double>{-0.5, 1.0});
        this->declare_parameter<std::vector<double>>("cmd_vel_limits.lin_vel_y",
            std::vector<double>{-0.5, 0.5});
        this->declare_parameter<std::vector<double>>("cmd_vel_limits.ang_vel_z",
            std::vector<double>{-1.0, 1.0});

        this->declare_parameter<std::vector<long int>>("upper_body_joint_indices",
            std::vector<long int>{13, 14, 15, 16, 18, 22, 23, 25});

        this->declare_parameter<std::vector<double>>("kp", std::vector<double>{
            40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
            40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
            40.2, 28.5, 28.5,
            14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
            14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8});

        this->declare_parameter<std::vector<double>>("kd", std::vector<double>{
            2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
            2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
            2.6, 1.8, 1.8,
            0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
            0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1});

        this->declare_parameter<std::vector<double>>("default_joint_pos", std::vector<double>{
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
             0.0, 0.0, 0.0,
             0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
             0.35,-0.18, 0.0, 0.87, 0.0, 0.0, 0.0});

        this->declare_parameter<std::vector<double>>("action_scale", std::vector<double>{
            0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
            0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
            0.35, 0.44, 0.44,
            0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,
            0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07});

        // Read parameters
        std::string model_path = this->get_parameter("model_path").as_string();
        control_dt_ = this->get_parameter("control_dt").as_double();
        double publish_dt = this->get_parameter("publish_dt").as_double();
        standup_duration_ = this->get_parameter("standup_duration").as_double();
        gait_period_ = this->get_parameter("gait_period").as_double();

        auto lim_x = this->get_parameter("cmd_vel_limits.lin_vel_x").as_double_array();
        auto lim_y = this->get_parameter("cmd_vel_limits.lin_vel_y").as_double_array();
        auto lim_z = this->get_parameter("cmd_vel_limits.ang_vel_z").as_double_array();
        vel_limit_x_ = {static_cast<float>(lim_x[0]), static_cast<float>(lim_x[1])};
        vel_limit_y_ = {static_cast<float>(lim_y[0]), static_cast<float>(lim_y[1])};
        vel_limit_z_ = {static_cast<float>(lim_z[0]), static_cast<float>(lim_z[1])};

        kp_ = this->get_parameter("kp").as_double_array();
        kd_ = this->get_parameter("kd").as_double_array();
        default_pos_ = this->get_parameter("default_joint_pos").as_double_array();
        action_scale_ = this->get_parameter("action_scale").as_double_array();

        // Initialize upper body command defaults from default_joint_pos
        for (int i = 0; i < NUM_UPPER_BODY; ++i)
            upper_body_cmd_[i] = static_cast<float>(default_pos_[UPPER_BODY_INDICES[i]]);

        // Build name -> upper body index map for /joint_commands lookup
        for (int i = 0; i < NUM_UPPER_BODY; ++i)
            upper_body_name_map_[UPPER_BODY_NAMES[i]] = i;

        // Load policy
        RCLCPP_INFO(this->get_logger(), "Loading policy: %s", model_path.c_str());
        policy_ = std::make_unique<OnnxPolicy>(model_path);
        last_action_.resize(NUM_MOTOR, 0.0f);

        RCLCPP_INFO(this->get_logger(), "control_dt=%.3f  publish_dt=%.3f  standup=%.1fs  gait_period=%.2fs",
                     control_dt_, publish_dt, standup_duration_, gait_period_);

        // QoS: best-effort + volatile to match Unitree SDK's DDS defaults
        auto sensor_qos = rclcpp::SensorDataQoS().keep_last(1);

        // Subscribers
        lowstate_sub_ = this->create_subscription<unitree_hg::msg::LowState>(
            "lowstate", sensor_qos,
            [this](const unitree_hg::msg::LowState::SharedPtr msg) { LowStateCallback(msg); });

        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", sensor_qos,
            [this](const geometry_msgs::msg::Twist::SharedPtr msg) { CmdVelCallback(msg); });

        joint_cmd_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_commands", sensor_qos,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) { JointCommandCallback(msg); });

        // Publisher
        lowcmd_pub_ = this->create_publisher<unitree_hg::msg::LowCmd>("/lowcmd", sensor_qos);

        // Timers
        control_timer_ = this->create_wall_timer(
            std::chrono::microseconds(static_cast<int>(control_dt_ * 1e6)),
            [this] { Control(); });

        publish_timer_ = this->create_wall_timer(
            std::chrono::microseconds(static_cast<int>(publish_dt * 1e6)),
            [this] { PublishCmd(); });

        RCLCPP_INFO(this->get_logger(), "Waiting for /lowstate, /cmd_vel, /joint_commands...");
    }

private:
    void CmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        vel_cmd_[0] = std::clamp(static_cast<float>(msg->linear.x), vel_limit_x_[0], vel_limit_x_[1]);
        vel_cmd_[1] = std::clamp(static_cast<float>(msg->linear.y), vel_limit_y_[0], vel_limit_y_[1]);
        vel_cmd_[2] = std::clamp(static_cast<float>(msg->angular.z), vel_limit_z_[0], vel_limit_z_[1]);
    }

    void JointCommandCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        for (size_t j = 0; j < msg->name.size() && j < msg->position.size(); ++j) {
            auto it = upper_body_name_map_.find(msg->name[j]);
            if (it != upper_body_name_map_.end())
                upper_body_cmd_[it->second] = static_cast<float>(msg->position[j]);
        }
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
            cmd.motor_cmd[i].kp = static_cast<float>(kp_[i]);
            cmd.motor_cmd[i].kd = static_cast<float>(kd_[i]);
            cmd.motor_cmd[i].tau = 0.0f;
            cmd.motor_cmd[i].dq = 0.0f;
        }

        time_ += control_dt_;

        if (time_ < standup_duration_) {
            float ratio = std::clamp(static_cast<float>(time_ / standup_duration_), 0.0f, 1.0f);
            for (int i = 0; i < NUM_MOTOR; ++i)
                cmd.motor_cmd[i].q = (1.0f - ratio) * motor_q_[i] + ratio * static_cast<float>(default_pos_[i]);

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

            // Build observation (106 dims)
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
            global_phase_ += static_cast<float>(control_dt_ / gait_period_);
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
                obs.push_back(motor_q_[i] - static_cast<float>(default_pos_[i]));

            // 6. joint_vel_rel (29)
            for (int i = 0; i < NUM_MOTOR; ++i)
                obs.push_back(motor_dq_[i]);

            // 7. last_action (29)
            for (int i = 0; i < NUM_MOTOR; ++i)
                obs.push_back(last_action_[i]);

            // 8. upper_body_command (8)
            for (int i = 0; i < NUM_UPPER_BODY; ++i)
                obs.push_back(upper_body_cmd_[i]);

            // Infer
            auto raw_action = policy_->infer(obs);
            last_action_ = raw_action;

            // target_q = raw * scale + offset
            for (int i = 0; i < NUM_MOTOR; ++i)
                cmd.motor_cmd[i].q = raw_action[i] * static_cast<float>(action_scale_[i])
                                   + static_cast<float>(default_pos_[i]);

            static int print_counter = 0;
            if (++print_counter % 250 == 0) {
                RCLCPP_INFO(this->get_logger(), "t=%.1f cmd_vel=[%.2f,%.2f,%.2f] ub=[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]",
                            time_, vel_cmd_[0], vel_cmd_[1], vel_cmd_[2],
                            upper_body_cmd_[0], upper_body_cmd_[1], upper_body_cmd_[2], upper_body_cmd_[3],
                            upper_body_cmd_[4], upper_body_cmd_[5], upper_body_cmd_[6], upper_body_cmd_[7]);
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
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_sub_;
    rclcpp::Publisher<unitree_hg::msg::LowCmd>::SharedPtr lowcmd_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::TimerBase::SharedPtr publish_timer_;

    // Policy
    std::unique_ptr<OnnxPolicy> policy_;
    std::vector<float> last_action_;

    // Parameters
    std::vector<double> kp_, kd_, default_pos_, action_scale_;
    double control_dt_, standup_duration_, gait_period_;
    std::array<float, 2> vel_limit_x_, vel_limit_y_, vel_limit_z_;

    // Upper body command
    std::array<float, NUM_UPPER_BODY> upper_body_cmd_ = {};
    std::unordered_map<std::string, int> upper_body_name_map_;

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
