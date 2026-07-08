#include <algorithm>
#include <cstddef>
#include <memory>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"

namespace {

constexpr std::size_t AXIS_LINEAR_Y = 0;
constexpr std::size_t AXIS_LINEAR_X = 1;
constexpr std::size_t AXIS_ANGULAR_Z = 3;

constexpr double LIN_X_SCALE = 2.0;
constexpr double LIN_Y_SCALE = 1.0;
constexpr double ANG_Z_SCALE = 1.0;

constexpr double LIN_X_MIN = -1.0;
constexpr double LIN_X_MAX = 2.0;
constexpr double LIN_Y_MIN = -1.0;
constexpr double LIN_Y_MAX = 1.0;
constexpr double ANG_Z_MIN = -1.0;
constexpr double ANG_Z_MAX = 1.0;

bool HasAxes(const sensor_msgs::msg::Joy &msg) {
  return msg.axes.size() > AXIS_ANGULAR_Z;
}

}  // namespace

class JoyCmdVelNode : public rclcpp::Node {
 public:
  JoyCmdVelNode() : Node("joy_cmd_vel_node") {
    auto qos = rclcpp::SensorDataQoS().keep_last(1);

    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", qos);
    joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", qos,
        [this](const sensor_msgs::msg::Joy::SharedPtr msg) { JoyCallback(*msg); });

    RCLCPP_INFO(this->get_logger(), "joy_cmd_vel_node started");
  }

 private:
  void JoyCallback(const sensor_msgs::msg::Joy &msg) {
    geometry_msgs::msg::Twist cmd;

    if (!HasAxes(msg)) {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Ignoring /joy velocity command: expected at least 4 axes, got %zu",
          msg.axes.size());
      cmd_vel_pub_->publish(cmd);
      return;
    }

    cmd.linear.x = std::clamp(
        static_cast<double>(msg.axes[AXIS_LINEAR_X]) * LIN_X_SCALE,
        LIN_X_MIN, LIN_X_MAX);
    cmd.linear.y = std::clamp(
        static_cast<double>(msg.axes[AXIS_LINEAR_Y]) * LIN_Y_SCALE,
        LIN_Y_MIN, LIN_Y_MAX);
    cmd.angular.z = std::clamp(
        static_cast<double>(msg.axes[AXIS_ANGULAR_Z]) * ANG_Z_SCALE,
        ANG_Z_MIN, ANG_Z_MAX);

    cmd_vel_pub_->publish(cmd);
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<JoyCmdVelNode>());
  rclcpp::shutdown();
  return 0;
}
