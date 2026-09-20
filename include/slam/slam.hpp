#pragma once

#include "research/math.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace slam {

struct Feature { research::Vec2 pixel; std::array<float, 32> descriptor{}; };
struct Match { std::size_t previous = 0; std::size_t current = 0; double distance = 0.0; };
struct Pose2 { double x = 0.0; double y = 0.0; double yaw = 0.0; };
struct MotionEstimate { Pose2 transform; std::vector<Match> inliers; double mean_residual = 0.0; };

[[nodiscard]] std::vector<Match> match_features(std::span<const Feature> previous,
                                                std::span<const Feature> current,
                                                double ratio_threshold = 0.75);
[[nodiscard]] MotionEstimate estimate_motion_ransac(std::span<const Feature> previous,
                                                    std::span<const Feature> current,
                                                    std::span<const Match> matches,
                                                    double inlier_threshold = 2.0);

struct PoseConstraint { std::size_t from = 0; std::size_t to = 0; Pose2 relative; double weight = 1.0; };

class PoseGraphOptimizer {
 public:
  [[nodiscard]] std::vector<Pose2> optimize(std::vector<Pose2> poses,
                                            std::span<const PoseConstraint> constraints,
                                            std::size_t iterations = 50,
                                            double learning_rate = 0.05) const;
};

}  // namespace slam
