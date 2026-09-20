#include "slam/opencv_features.hpp"

#include <opencv2/features2d.hpp>

namespace slam {
std::vector<Feature> extract_orb_features(const cv::Mat& image, const std::size_t maximum_features) {
  auto detector = cv::ORB::create(static_cast<int>(maximum_features));
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
  std::vector<Feature> output;
  output.reserve(keypoints.size());
  for (std::size_t row = 0; row < keypoints.size(); ++row) {
    Feature feature;
    feature.pixel = {keypoints[row].pt.x, keypoints[row].pt.y};
    for (std::size_t column = 0; column < feature.descriptor.size(); ++column)
      feature.descriptor[column] = static_cast<float>(descriptors.at<unsigned char>(static_cast<int>(row), static_cast<int>(column))) / 255.0F;
    output.push_back(feature);
  }
  return output;
}
}  // namespace slam
