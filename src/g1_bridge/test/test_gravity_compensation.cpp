#include "g1_bridge/gravity_compensation.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

constexpr char TEST_URDF[] = G1_BRIDGE_TEST_URDF;

TEST(GravityCompensation, ComputesPendulumHoldingTorque) {
  g1_bridge::GravityCompensation gravity(TEST_URDF, {"shoulder_joint"});

  const auto torque = gravity.Compute({1.0, 0.0, 0.0, 0.0}, {0.0});

  ASSERT_EQ(torque.size(), 1U);
  EXPECT_NEAR(torque[0], -9.81, 1e-9);
}

TEST(GravityCompensation, UsesBaseOrientationAndNormalizesQuaternion) {
  g1_bridge::GravityCompensation gravity(TEST_URDF, {"shoulder_joint"});
  constexpr double base_pitch = 0.4;
  constexpr double joint_angle = 0.1;
  const double half_pitch = base_pitch / 2.0;

  const auto torque = gravity.Compute(
      {2.0 * std::cos(half_pitch), 0.0,
       2.0 * std::sin(half_pitch), 0.0},
      {joint_angle});

  EXPECT_NEAR(torque[0], -9.81 * std::cos(base_pitch + joint_angle), 1e-9);
}

TEST(GravityCompensation, RejectsInvalidState) {
  g1_bridge::GravityCompensation gravity(TEST_URDF, {"shoulder_joint"});

  EXPECT_THROW(gravity.Compute({0.0, 0.0, 0.0, 0.0}, {0.0}),
               std::invalid_argument);
  EXPECT_THROW(gravity.Compute({1.0, 0.0, 0.0, 0.0}, {}),
               std::invalid_argument);
}

}  // namespace
