#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include <Eigen/Dense>

// ONNX Runtime
#include "onnxruntime_cxx_api.h"

// Unitree SDK2 DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

// ---------------------------------------------------------------------------
// Constants from deploy.yaml (Unitree-G1-Flat velocity policy, run1)
// ---------------------------------------------------------------------------
static constexpr int NUM_MOTOR = 29;
static constexpr int OBS_DIM = 98;
static constexpr float STEP_DT = 0.02f;        // 50 Hz policy
static constexpr float CONTROL_DT = 0.002f;    // 500 Hz command writer
static constexpr float STANDUP_DURATION = 3.0f; // seconds
static constexpr float GAIT_PERIOD = 0.6f;

static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_STATE_TOPIC = "rt/lowstate";

// Per-joint stiffness (kp)
static constexpr std::array<float, NUM_MOTOR> KP = {
    40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f,   // left leg
    40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f,   // right leg
    40.2f, 28.5f, 28.5f,                          // waist
    14.3f, 14.3f, 14.3f, 14.3f, 14.3f, 16.8f, 16.8f, // left arm
    14.3f, 14.3f, 14.3f, 14.3f, 14.3f, 16.8f, 16.8f  // right arm
};

// Per-joint damping (kd)
static constexpr std::array<float, NUM_MOTOR> KD = {
    2.6f, 6.3f, 2.6f, 6.3f, 1.8f, 1.8f,
    2.6f, 6.3f, 2.6f, 6.3f, 1.8f, 1.8f,
    2.6f, 1.8f, 1.8f,
    0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 1.1f, 1.1f,
    0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 1.1f, 1.1f
};

// Default joint positions (home pose / action offset)
static constexpr std::array<float, NUM_MOTOR> DEFAULT_POS = {
    -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,      // left leg
    -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,      // right leg
     0.0f, 0.0f, 0.0f,                           // waist
     0.35f, 0.18f, 0.0f, 0.87f, 0.0f, 0.0f, 0.0f, // left arm
     0.35f,-0.18f, 0.0f, 0.87f, 0.0f, 0.0f, 0.0f  // right arm
};

// Action scale
static constexpr std::array<float, NUM_MOTOR> ACTION_SCALE = {
    0.55f, 0.35f, 0.55f, 0.35f, 0.44f, 0.44f,
    0.55f, 0.35f, 0.55f, 0.35f, 0.44f, 0.44f,
    0.35f, 0.44f, 0.44f,
    0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.07f, 0.07f,
    0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.07f, 0.07f
};

// Hardcoded velocity command for testing (forward walk)
static constexpr std::array<float, 3> VEL_CMD = {0.5f, 0.0f, 0.0f};

// ---------------------------------------------------------------------------
// Thread-safe data buffer (from SDK example)
// ---------------------------------------------------------------------------
template <typename T>
class DataBuffer {
public:
    void SetData(const T& newData) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        data_ = std::make_shared<T>(newData);
    }
    std::shared_ptr<const T> GetData() {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return data_ ? data_ : nullptr;
    }
private:
    std::shared_ptr<T> data_;
    std::shared_mutex mutex_;
};

// ---------------------------------------------------------------------------
// Motor / IMU state structs
// ---------------------------------------------------------------------------
struct MotorState {
    std::array<float, NUM_MOTOR> q = {};
    std::array<float, NUM_MOTOR> dq = {};
};

struct ImuData {
    std::array<float, 4> quaternion = {};  // w, x, y, z
    std::array<float, 3> gyroscope = {};
};

struct MotorCommand {
    std::array<float, NUM_MOTOR> q_target = {};
    std::array<float, NUM_MOTOR> dq_target = {};
    std::array<float, NUM_MOTOR> kp = {};
    std::array<float, NUM_MOTOR> kd = {};
    std::array<float, NUM_MOTOR> tau_ff = {};
};

// ---------------------------------------------------------------------------
// CRC32 (from SDK example)
// ---------------------------------------------------------------------------
inline uint32_t Crc32Core(uint32_t* ptr, uint32_t len) {
    uint32_t xbit = 0;
    uint32_t data = 0;
    uint32_t CRC32 = 0xFFFFFFFF;
    const uint32_t dwPolynomial = 0x04c11db7;
    for (uint32_t i = 0; i < len; i++) {
        xbit = 1 << 31;
        data = ptr[i];
        for (uint32_t bits = 0; bits < 32; bits++) {
            if (CRC32 & 0x80000000) {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            } else {
                CRC32 <<= 1;
            }
            if (data & xbit) CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }
    return CRC32;
}

// ---------------------------------------------------------------------------
// ONNX Policy wrapper
// ---------------------------------------------------------------------------
class OnnxPolicy {
public:
    OnnxPolicy(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "g1_policy") {
        session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

        // Query input info
        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            auto type_info = session_->GetInputTypeInfo(i);
            input_shapes_.push_back(type_info.GetTensorTypeAndShapeInfo().GetShape());
            auto name = session_->GetInputNameAllocated(i, allocator_);
            input_name_strs_.push_back(name.get());

            size_t size = 1;
            for (auto dim : input_shapes_.back()) size *= dim;
            input_sizes_.push_back(size);
        }
        // Build char* array after all strings are in place (avoids reallocation invalidation)
        for (auto& s : input_name_strs_)
            input_names_.push_back(s.c_str());

        // Query output info
        auto out_type = session_->GetOutputTypeInfo(0);
        output_shape_ = out_type.GetTensorTypeAndShapeInfo().GetShape();
        auto out_name = session_->GetOutputNameAllocated(0, allocator_);
        output_name_str_ = out_name.get();
        output_name_ = output_name_str_.c_str();

        std::cout << "[OnnxPolicy] Loaded model: " << model_path << std::endl;
        std::cout << "[OnnxPolicy] Input: \"" << input_name_strs_[0]
                  << "\" shape=[" << input_shapes_[0][0] << "," << input_shapes_[0][1] << "]" << std::endl;
        std::cout << "[OnnxPolicy] Output shape=[" << output_shape_[0] << "," << output_shape_[1] << "]" << std::endl;
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
        int output_size = static_cast<int>(output_shape_[1]);
        return std::vector<float>(floatarr, floatarr + output_size);
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
// G1 RL Deploy
// ---------------------------------------------------------------------------
class G1RLDeploy {
public:
    G1RLDeploy(const std::string& network_interface, const std::string& model_path)
        : time_(0.0),
          policy_time_(0.0),
          global_phase_(0.0f),
          mode_machine_(0),
          running_policy_(false) {

        // Initialize ONNX policy
        policy_ = std::make_unique<OnnxPolicy>(model_path);
        last_action_.resize(NUM_MOTOR, 0.0f);

        // Initialize DDS
        ChannelFactory::Instance()->Init(0, network_interface);

        lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
        lowcmd_publisher_->InitChannel();

        lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
        lowstate_subscriber_->InitChannel(
            std::bind(&G1RLDeploy::LowStateHandler, this, std::placeholders::_1), 1);

        std::cout << "[G1RLDeploy] DDS initialized on interface: " << network_interface << std::endl;
        std::cout << "[G1RLDeploy] Waiting for first LowState message..." << std::endl;

        // Create threads: command writer at 500 Hz, control at 50 Hz
        command_writer_ptr_ = CreateRecurrentThreadEx(
            "command_writer", UT_CPU_ID_NONE, 2000, &G1RLDeploy::LowCommandWriter, this);
        control_thread_ptr_ = CreateRecurrentThreadEx(
            "control", UT_CPU_ID_NONE, 20000, &G1RLDeploy::Control, this);
    }

private:
    void LowStateHandler(const void* message) {
        LowState_ low_state = *(const LowState_*)message;

        // Extract motor state
        MotorState ms;
        for (int i = 0; i < NUM_MOTOR; ++i) {
            ms.q[i] = low_state.motor_state()[i].q();
            ms.dq[i] = low_state.motor_state()[i].dq();
        }
        motor_state_buffer_.SetData(ms);

        // Extract IMU
        ImuData imu;
        for (int i = 0; i < 4; ++i)
            imu.quaternion[i] = low_state.imu_state().quaternion()[i];
        for (int i = 0; i < 3; ++i)
            imu.gyroscope[i] = low_state.imu_state().gyroscope()[i];
        imu_buffer_.SetData(imu);

        // Track mode_machine
        if (mode_machine_ != low_state.mode_machine()) {
            if (mode_machine_ == 0)
                std::cout << "[G1RLDeploy] G1 type: " << unsigned(low_state.mode_machine()) << std::endl;
            mode_machine_ = low_state.mode_machine();
        }
    }

    void Control() {
        auto ms = motor_state_buffer_.GetData();
        auto imu = imu_buffer_.GetData();
        if (!ms || !imu) return;

        MotorCommand cmd;
        for (int i = 0; i < NUM_MOTOR; ++i) {
            cmd.kp[i] = KP[i];
            cmd.kd[i] = KD[i];
            cmd.tau_ff[i] = 0.0f;
            cmd.dq_target[i] = 0.0f;
        }

        time_ += STEP_DT;

        if (time_ < STANDUP_DURATION) {
            // Phase 1: interpolate to default pose
            float ratio = std::clamp(static_cast<float>(time_ / STANDUP_DURATION), 0.0f, 1.0f);
            for (int i = 0; i < NUM_MOTOR; ++i) {
                cmd.q_target[i] = (1.0f - ratio) * ms->q[i] + ratio * DEFAULT_POS[i];
            }
            if (!running_policy_) {
                static bool printed = false;
                if (!printed) {
                    std::cout << "[G1RLDeploy] Phase 1: Standing up (" << STANDUP_DURATION << "s)..." << std::endl;
                    printed = true;
                }
            }
        } else {
            // Phase 2: run policy
            if (!running_policy_) {
                running_policy_ = true;
                std::cout << "[G1RLDeploy] Phase 2: Policy active (forward vel=" << VEL_CMD[0] << " m/s)" << std::endl;
            }

            // Build observation vector (98 dims)
            std::vector<float> obs;
            obs.reserve(OBS_DIM);

            // 1. base_ang_vel (3)
            for (int i = 0; i < 3; ++i)
                obs.push_back(imu->gyroscope[i]);

            // 2. projected_gravity (3)
            Eigen::Quaternionf q(imu->quaternion[0], imu->quaternion[1],
                                 imu->quaternion[2], imu->quaternion[3]);
            Eigen::Vector3f gravity_b = q.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
            obs.push_back(gravity_b.x());
            obs.push_back(gravity_b.y());
            obs.push_back(gravity_b.z());

            // 3. velocity_commands (3)
            for (int i = 0; i < 3; ++i)
                obs.push_back(VEL_CMD[i]);

            // 4. gait_phase (2)
            float delta_phase = STEP_DT * (1.0f / GAIT_PERIOD);
            global_phase_ += delta_phase;
            global_phase_ = std::fmod(global_phase_, 1.0f);

            float cmd_norm = std::sqrt(VEL_CMD[0]*VEL_CMD[0] + VEL_CMD[1]*VEL_CMD[1] + VEL_CMD[2]*VEL_CMD[2]);
            if (cmd_norm < 0.1f) {
                obs.push_back(0.0f);
                obs.push_back(0.0f);
            } else {
                obs.push_back(std::sin(global_phase_ * 2.0f * M_PI));
                obs.push_back(std::cos(global_phase_ * 2.0f * M_PI));
            }

            // 5. joint_pos_rel (29)
            for (int i = 0; i < NUM_MOTOR; ++i)
                obs.push_back(ms->q[i] - DEFAULT_POS[i]);

            // 6. joint_vel_rel (29)
            for (int i = 0; i < NUM_MOTOR; ++i)
                obs.push_back(ms->dq[i]);

            // 7. last_action (29)
            for (int i = 0; i < NUM_MOTOR; ++i)
                obs.push_back(last_action_[i]);

            // Run ONNX inference
            auto raw_action = policy_->infer(obs);
            last_action_ = raw_action;

            // Process action: target_q = raw * scale + offset
            for (int i = 0; i < NUM_MOTOR; ++i) {
                cmd.q_target[i] = raw_action[i] * ACTION_SCALE[i] + DEFAULT_POS[i];
            }

            // Print status periodically
            policy_time_ += STEP_DT;
            if (static_cast<int>(policy_time_ / STEP_DT) % 250 == 0) {
                printf("[G1RLDeploy] t=%.1fs  q0=%.3f  q3=%.3f  q6=%.3f  q9=%.3f\n",
                       time_, cmd.q_target[0], cmd.q_target[3], cmd.q_target[6], cmd.q_target[9]);
            }
        }

        motor_command_buffer_.SetData(cmd);
    }

    void LowCommandWriter() {
        auto mc = motor_command_buffer_.GetData();
        if (!mc) return;

        LowCmd_ dds_cmd;
        dds_cmd.mode_pr() = 0;  // PR mode (pitch/roll)
        dds_cmd.mode_machine() = mode_machine_;

        for (int i = 0; i < NUM_MOTOR; ++i) {
            dds_cmd.motor_cmd()[i].mode() = 1;  // enabled
            dds_cmd.motor_cmd()[i].q() = mc->q_target[i];
            dds_cmd.motor_cmd()[i].dq() = mc->dq_target[i];
            dds_cmd.motor_cmd()[i].kp() = mc->kp[i];
            dds_cmd.motor_cmd()[i].kd() = mc->kd[i];
            dds_cmd.motor_cmd()[i].tau() = mc->tau_ff[i];
        }

        dds_cmd.crc() = Crc32Core((uint32_t*)&dds_cmd, (sizeof(dds_cmd) >> 2) - 1);
        lowcmd_publisher_->Write(dds_cmd);
    }

    // State
    double time_;
    double policy_time_;
    float global_phase_;
    uint8_t mode_machine_;
    bool running_policy_;

    // Policy
    std::unique_ptr<OnnxPolicy> policy_;
    std::vector<float> last_action_;

    // Buffers
    DataBuffer<MotorState> motor_state_buffer_;
    DataBuffer<ImuData> imu_buffer_;
    DataBuffer<MotorCommand> motor_command_buffer_;

    // DDS
    ChannelPublisherPtr<LowCmd_> lowcmd_publisher_;
    ChannelSubscriberPtr<LowState_> lowstate_subscriber_;
    ThreadPtr command_writer_ptr_, control_thread_ptr_;
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char const* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: g1_rl_deploy <network_interface> [model_path]" << std::endl;
        std::cout << "  network_interface: e.g. 'lo' for simulation, 'eth0' for real robot" << std::endl;
        std::cout << "  model_path: path to policy.onnx (default: ../policy.onnx)" << std::endl;
        return 1;
    }

    std::string network_interface = argv[1];
    std::string model_path = (argc >= 3) ? argv[2] : "../policy.onnx";

    std::cout << "=== G1 RL Deploy ===" << std::endl;
    std::cout << "Network: " << network_interface << std::endl;
    std::cout << "Model: " << model_path << std::endl;

    G1RLDeploy deploy(network_interface, model_path);

    while (true) sleep(10);
    return 0;
}
