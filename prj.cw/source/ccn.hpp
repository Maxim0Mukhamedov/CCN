#ifndef COMPR_COLOR_IMAGE_NORM_HPP
#define COMPR_COLOR_IMAGE_NORM_HPP

/**
 * @file ccn.hpp
 * @brief This file contains the declaration of the CCN algorithm.
 */

#include "vector"

namespace cv {
    class Mat;
}

namespace ccn {
    bool CheckImageSizeAndType(const cv::Mat &img1, const cv::Mat &img2);
    cv::Mat CopyLightness(cv::Mat newImg, cv::Mat originImg);
    double CalculateAverageDiff(const cv::Mat &img1, const cv::Mat &img2);
    void GrayWorldAssumption(cv::Mat &img);
    void NormalizeColors(cv::Mat &img);
    cv::Mat ComprColorImageNorm(const cv::Mat &img, const double &convValue = 0.1, const bool &CL = true);
    cv::Mat CalculateDiff(const cv::Mat &img1, const cv::Mat &img2);
    std::vector<cv::Mat> ComprColorImageNormDiff(const cv::Mat &img, const double &lr = 1);
}
#endif // COMPR_COLOR_IMAGE_NORM_HPP