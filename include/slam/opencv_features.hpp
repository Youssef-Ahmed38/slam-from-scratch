#pragma once

#include "slam/slam.hpp"

#if !defined(RESEARCH_HAS_OPENCV)
#error "Build with RESEARCH_ENABLE_OPENCV=ON to use this adapter"
#endif

#include <opencv2/core.hpp>

namespace slam {
[[nodiscard]] std::vector<Feature> extract_orb_features(const cv::Mat& image,
                                                        std::size_t maximum_features = 1000);
}  // namespace slam
