#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>

#include "g1_bridge/unitree_link.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rosgraph_msgs/msg/clock.hpp"

namespace {

constexpr char LOOPBACK_INTERFACE[] = "lo";

// Lets at most rate_hz messages per second of simulation time through. A
// stamp that jumps backwards, as when the simulator resets, passes at once.
class SimTimeRateLimiter {
 public:
  explicit SimTimeRateLimiter(double rate_hz)
      : period_ns_(rate_hz > 0.0 ? static_cast<std::int64_t>(1e9 / rate_hz)
                                 : 0) {}

  bool Due(const g1_bridge::TimeStamp &stamp) {
    const std::int64_t now_ns = stamp.sec * 1000000000LL + stamp.nanosec;
    if (now_ns < next_ns_ && now_ns + period_ns_ >= next_ns_) {
      return false;
    }
    next_ns_ = now_ns + period_ns_;
    return true;
  }

 private:
  std::int64_t period_ns_;
  std::int64_t next_ns_{0};
};

}  // namespace

// unitree_mujoco publishes simulation time and the pelvis pose on DDS only
// (rt/clock, rt/pose/pelvis). This node republishes them as /clock and
// /pose/pelvis. It only listens: it writes no rt/lowcmd and makes no motion
// service calls, so it can run before and independently of g1_bridge_node.
// It is the source of /clock and must not itself run on simulation time.
class SimRelayNode : public rclcpp::Node {
 public:
  SimRelayNode() : Node("sim_relay_node") {
    const auto network_interface =
        this->declare_parameter<std::string>("network_interface",
                                             LOOPBACK_INTERFACE);
    const auto domain_id =
        static_cast<int>(this->declare_parameter<int>("domain_id", 0));
    // Every node on simulation time wakes on each /clock, and every TF
    // listener on each transform made from /pose/pelvis. Forwarding them at
    // the simulator's full rate saturates the Jetson. 0 forwards everything.
    // /clock has to stay above the fastest ROS timer on simulation time, or
    // that timer slows down (the bridge's lowcmd timer runs on wall time).
    clock_limiter_ = std::make_unique<SimTimeRateLimiter>(
        this->declare_parameter<double>("clock_rate", 250.0));
    pelvis_pose_limiter_ = std::make_unique<SimTimeRateLimiter>(
        this->declare_parameter<double>("pelvis_pose_rate", 100.0));

    clock_pub_ = this->create_publisher<rosgraph_msgs::msg::Clock>(
        "/clock", rclcpp::ClockQoS());
    pelvis_pose_pub_ =
        this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/pose/pelvis", rclcpp::QoS(rclcpp::KeepLast(1))
                                .best_effort()
                                .durability_volatile());

    // The flag is shared, so the callback stays valid after this node is
    // gone. Set before the RMW shuts down, unlike rclcpp::ok().
    this->get_node_base_interface()->get_context()->add_pre_shutdown_callback(
        [shutting_down = ros_shutting_down_] { shutting_down->store(true); });

    RCLCPP_INFO(this->get_logger(),
                "Listening to the simulator with unitree_sdk2 on network "
                "interface '%s', DDS domain %d",
                network_interface.c_str(), domain_id);
    link_ = std::make_unique<g1_bridge::UnitreeLink>(domain_id,
                                                     network_interface);
    link_->SubscribeClock(
        [this](const g1_bridge::TimeStamp &stamp) { ClockCallback(stamp); });
    link_->SubscribePelvisPose(
        [this](const g1_bridge::StampedPose &pose) {
          PelvisPoseCallback(pose);
        });

    RCLCPP_INFO(this->get_logger(), "Simulator relay node started");
  }

 private:
  // Runs on an SDK thread.
  void ClockCallback(const g1_bridge::TimeStamp &stamp) {
    if (ros_shutting_down_->load() || !clock_limiter_->Due(stamp)) {
      return;
    }
    try {
      rosgraph_msgs::msg::Clock msg;
      msg.clock.sec = stamp.sec;
      msg.clock.nanosec = stamp.nanosec;
      clock_pub_->publish(msg);
    } catch (const std::exception &e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), steady_clock_, 2000,
                            "Failed to forward the simulator clock: %s",
                            e.what());
    }
  }

  // Runs on an SDK thread.
  void PelvisPoseCallback(const g1_bridge::StampedPose &pose) {
    if (ros_shutting_down_->load() || !pelvis_pose_limiter_->Due(pose.stamp)) {
      return;
    }
    try {
      geometry_msgs::msg::PoseStamped msg;
      msg.header.stamp.sec = pose.stamp.sec;
      msg.header.stamp.nanosec = pose.stamp.nanosec;
      msg.header.frame_id = pose.frame_id;
      msg.pose.position.x = pose.position[0];
      msg.pose.position.y = pose.position[1];
      msg.pose.position.z = pose.position[2];
      msg.pose.orientation.x = pose.orientation_xyzw[0];
      msg.pose.orientation.y = pose.orientation_xyzw[1];
      msg.pose.orientation.z = pose.orientation_xyzw[2];
      msg.pose.orientation.w = pose.orientation_xyzw[3];
      pelvis_pose_pub_->publish(msg);
    } catch (const std::exception &e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), steady_clock_, 2000,
                            "Failed to forward the simulator pelvis pose: %s",
                            e.what());
    }
  }

  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
      pelvis_pose_pub_;
  // Each is used by one SDK thread.
  std::unique_ptr<SimTimeRateLimiter> clock_limiter_;
  std::unique_ptr<SimTimeRateLimiter> pelvis_pose_limiter_;
  std::shared_ptr<std::atomic_bool> ros_shutting_down_{
      std::make_shared<std::atomic_bool>(false)};
  rclcpp::Clock steady_clock_{RCL_STEADY_TIME};

  // Declared last so that it is destroyed first: its destructor stops the SDK
  // threads that call back into the members above.
  std::unique_ptr<g1_bridge::UnitreeLink> link_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  int ret = 0;
  try {
    rclcpp::spin(std::make_shared<SimRelayNode>());
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("sim_relay_node"), "%s", e.what());
    ret = 1;
  }
  rclcpp::shutdown();
  return ret;
}
