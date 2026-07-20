#include "g1_bridge/gravity_compensation.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/joint/joint-free-flyer.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace g1_bridge {

namespace {

struct JointIndex {
  int configuration{};
  int velocity{};
};

}  // namespace

class GravityCompensation::Impl {
 public:
  Impl(const std::string &urdf_path,
       const std::vector<std::string> &joint_names) {
    if (urdf_path.empty()) {
      throw std::invalid_argument(
          "urdf_path must be set when gravity compensation is enabled");
    }
    if (joint_names.empty()) {
      throw std::invalid_argument("joint_names must not be empty");
    }

    pinocchio::urdf::buildModel(
        urdf_path, pinocchio::JointModelFreeFlyer(), model_);
    if (model_.njoints < 2 || model_.joints[1].nq() != 7 ||
        model_.joints[1].nv() != 6) {
      throw std::runtime_error(
          "gravity model does not have the expected free-flyer root joint");
    }

    root_configuration_ = model_.joints[1].idx_q();
    configuration_ = pinocchio::neutral(model_);
    data_ = std::make_unique<pinocchio::Data>(model_);

    joint_indices_.reserve(joint_names.size());
    for (const auto &joint_name : joint_names) {
      if (!model_.existJointName(joint_name)) {
        throw std::runtime_error(
            "gravity model is missing joint '" + joint_name + "'");
      }
      const auto joint_id = model_.getJointId(joint_name);
      const auto &joint = model_.joints[joint_id];
      if (joint.nq() != 1 || joint.nv() != 1) {
        throw std::runtime_error(
            "gravity model joint '" + joint_name +
            "' is not a scalar joint");
      }
      joint_indices_.push_back({joint.idx_q(), joint.idx_v()});
    }
    gravity_.resize(joint_indices_.size());
  }

  const std::vector<double> &Compute(
      const std::array<double, 4> &base_orientation_wxyz,
      const std::vector<double> &joint_positions) {
    if (joint_positions.size() != joint_indices_.size()) {
      throw std::invalid_argument(
          "joint_positions size does not match the gravity model joint order");
    }

    double squared_norm = 0.0;
    for (const double value : base_orientation_wxyz) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "base orientation quaternion must contain finite values");
      }
      squared_norm += value * value;
    }
    if (!std::isfinite(squared_norm) ||
        squared_norm <= std::numeric_limits<double>::min()) {
      throw std::invalid_argument(
          "base orientation quaternion must be nonzero");
    }

    const double inverse_norm = 1.0 / std::sqrt(squared_norm);
    // Pinocchio's free-flyer configuration stores its quaternion as xyzw,
    // while Unitree LowState and this API use wxyz.
    configuration_[root_configuration_ + 3] =
        base_orientation_wxyz[1] * inverse_norm;
    configuration_[root_configuration_ + 4] =
        base_orientation_wxyz[2] * inverse_norm;
    configuration_[root_configuration_ + 5] =
        base_orientation_wxyz[3] * inverse_norm;
    configuration_[root_configuration_ + 6] =
        base_orientation_wxyz[0] * inverse_norm;

    for (std::size_t i = 0; i < joint_positions.size(); ++i) {
      if (!std::isfinite(joint_positions[i])) {
        throw std::invalid_argument(
            "joint_positions must contain finite values");
      }
      configuration_[joint_indices_[i].configuration] = joint_positions[i];
    }

    const auto &generalized_gravity = pinocchio::computeGeneralizedGravity(
        model_, *data_, configuration_);
    for (std::size_t i = 0; i < joint_indices_.size(); ++i) {
      gravity_[i] = generalized_gravity[joint_indices_[i].velocity];
    }
    return gravity_;
  }

  std::size_t joint_count() const noexcept { return joint_indices_.size(); }

 private:
  pinocchio::Model model_;
  std::unique_ptr<pinocchio::Data> data_;
  Eigen::VectorXd configuration_;
  std::vector<JointIndex> joint_indices_;
  std::vector<double> gravity_;
  int root_configuration_{};
};

GravityCompensation::GravityCompensation(
    const std::string &urdf_path,
    const std::vector<std::string> &joint_names)
    : impl_(std::make_unique<Impl>(urdf_path, joint_names)) {}

GravityCompensation::~GravityCompensation() = default;
GravityCompensation::GravityCompensation(GravityCompensation &&) noexcept =
    default;
GravityCompensation &GravityCompensation::operator=(
    GravityCompensation &&) noexcept = default;

const std::vector<double> &GravityCompensation::Compute(
    const std::array<double, 4> &base_orientation_wxyz,
    const std::vector<double> &joint_positions) {
  return impl_->Compute(base_orientation_wxyz, joint_positions);
}

std::size_t GravityCompensation::joint_count() const noexcept {
  return impl_->joint_count();
}

}  // namespace g1_bridge
