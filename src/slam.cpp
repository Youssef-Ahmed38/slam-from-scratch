#include "slam/slam.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace slam {
namespace {
double descriptor_distance(const Feature& left, const Feature& right) { double sum = 0.0; for (std::size_t i = 0; i < left.descriptor.size(); ++i) { const double delta = static_cast<double>(left.descriptor[i] - right.descriptor[i]); sum += delta * delta; } return std::sqrt(sum); }
research::Vec2 transform(research::Vec2 point, const Pose2& pose) { const double cosine = std::cos(pose.yaw); const double sine = std::sin(pose.yaw); return {cosine * point.x - sine * point.y + pose.x, sine * point.x + cosine * point.y + pose.y}; }
Pose2 fit_rigid(std::span<const Feature> previous, std::span<const Feature> current, std::span<const Match> matches) {
  if (matches.size() < 2) throw std::invalid_argument("at least two matches are required"); research::Vec2 a{}, b{}; for (const auto& match : matches) { a = a + previous[match.previous].pixel; b = b + current[match.current].pixel; } a = a * (1.0 / static_cast<double>(matches.size())); b = b * (1.0 / static_cast<double>(matches.size())); double cross = 0.0; double direct = 0.0; for (const auto& match : matches) { const auto p = previous[match.previous].pixel - a; const auto q = current[match.current].pixel - b; cross += p.x * q.y - p.y * q.x; direct += p.x * q.x + p.y * q.y; } const double yaw = std::atan2(cross, direct); const auto rotated = transform(a, {0.0, 0.0, yaw}); return {b.x - rotated.x, b.y - rotated.y, yaw};
}
}

std::vector<Match> match_features(const std::span<const Feature> previous, const std::span<const Feature> current, const double ratio) {
  std::vector<Match> matches; matches.reserve(std::min(previous.size(), current.size()));
  for (std::size_t i = 0; i < previous.size(); ++i) { double best = std::numeric_limits<double>::infinity(); double second = best; std::size_t best_index = 0; for (std::size_t j = 0; j < current.size(); ++j) { const double distance = descriptor_distance(previous[i], current[j]); if (distance < best) { second = best; best = distance; best_index = j; } else if (distance < second) second = distance; } if (best < ratio * second) matches.push_back({i, best_index, best}); }
  return matches;
}

MotionEstimate estimate_motion_ransac(const std::span<const Feature> previous, const std::span<const Feature> current, const std::span<const Match> matches, const double threshold) {
  if (matches.size() < 2) throw std::invalid_argument("insufficient feature matches"); std::vector<Match> best;
  const std::size_t trials = std::min<std::size_t>(matches.size() * (matches.size() - 1) / 2, 256); std::size_t trial = 0;
  for (std::size_t first = 0; first < matches.size() && trial < trials; ++first) for (std::size_t second = first + 1; second < matches.size() && trial < trials; ++second, ++trial) { const std::array<Match, 2> sample{matches[first], matches[second]}; const auto pose = fit_rigid(previous, current, sample); std::vector<Match> inliers; for (const auto& match : matches) if (research::norm(transform(previous[match.previous].pixel, pose) - current[match.current].pixel) <= threshold) inliers.push_back(match); if (inliers.size() > best.size()) best = std::move(inliers); }
  if (best.size() < 2) throw std::runtime_error("RANSAC found no valid motion"); const auto pose = fit_rigid(previous, current, best); double residual = 0.0; for (const auto& match : best) residual += research::norm(transform(previous[match.previous].pixel, pose) - current[match.current].pixel); return {pose, std::move(best), residual / static_cast<double>(best.size())};
}

std::vector<Pose2> PoseGraphOptimizer::optimize(std::vector<Pose2> poses, const std::span<const PoseConstraint> constraints, const std::size_t iterations, const double rate) const {
  if (poses.empty()) return poses; for (const auto& edge : constraints) if (edge.from >= poses.size() || edge.to >= poses.size()) throw std::out_of_range("pose constraint index");
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) { std::vector<Pose2> gradients(poses.size()); for (const auto& edge : constraints) { const Pose2 prediction{poses[edge.to].x - poses[edge.from].x, poses[edge.to].y - poses[edge.from].y, poses[edge.to].yaw - poses[edge.from].yaw}; const Pose2 error{prediction.x - edge.relative.x, prediction.y - edge.relative.y, prediction.yaw - edge.relative.yaw}; gradients[edge.to].x += edge.weight * error.x; gradients[edge.to].y += edge.weight * error.y; gradients[edge.to].yaw += edge.weight * error.yaw; if (edge.from != 0) { gradients[edge.from].x -= edge.weight * error.x; gradients[edge.from].y -= edge.weight * error.y; gradients[edge.from].yaw -= edge.weight * error.yaw; } } for (std::size_t i = 1; i < poses.size(); ++i) { poses[i].x -= rate * gradients[i].x; poses[i].y -= rate * gradients[i].y; poses[i].yaw -= rate * gradients[i].yaw; } }
  return poses;
}
}  // namespace slam
