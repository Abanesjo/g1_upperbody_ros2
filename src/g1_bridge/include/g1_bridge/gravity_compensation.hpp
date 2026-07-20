#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace g1_bridge {

// Computes model gravity in the caller-provided joint order. The rigid-body
// model is intentionally hidden so users of this header do not depend on
// Pinocchio implementation details.
class GravityCompensation {
 public:
  GravityCompensation(const std::string &urdf_path,
                      const std::vector<std::string> &joint_names);
  ~GravityCompensation();

  GravityCompensation(GravityCompensation &&) noexcept;
  GravityCompensation &operator=(GravityCompensation &&) noexcept;

  GravityCompensation(const GravityCompensation &) = delete;
  GravityCompensation &operator=(const GravityCompensation &) = delete;

  const std::vector<double> &Compute(
      const std::array<double, 4> &base_orientation_wxyz,
      const std::vector<double> &joint_positions);

  std::size_t joint_count() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace g1_bridge
