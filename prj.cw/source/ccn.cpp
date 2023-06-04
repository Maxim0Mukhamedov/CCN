/**
 * @file ccn.cpp
 * @brief This file contains the realisation of the CCN algorithm.
 */

#include <opencv2/core.hpp>
#include "stdexcept"
#include "vector"
#include "ccn.hpp"
#include <cmath>

namespace vision {
// Convertes an sRGB value in the range [0, 255] to a linear value in
// the range [0, 1].
    inline float srgb_to_linear(float srgb) {
        auto linear = srgb / 255.0f;
        if (linear <= 0.04045f) {
            linear = linear / 12.92f;
        } else {
            linear = powf((linear + 0.055f) / 1.055f, 2.4f);
        }
        return linear;
    }

// Converts a linear value in the range [0, 1] to an sRGB value in
// the range [0, 255].
    inline float linear_to_srgb(float linear) {
        float srgb;
        if (linear <= 0.0031308f) {
            srgb = linear * 12.92f;
        } else {
            srgb = 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
        }
        return srgb * 255.f;
    }

// Converts an OpenCV Mat with sRGB values in the range [0, 255] to
// a float (CV_32FC3) OpenCV Mat in the range [0, 1].
    static cv::Mat srgb_to_linear(const cv::Mat& mat) {
        cv::Mat linear;
        mat.convertTo(linear, CV_32FC3);
        for (auto itr = linear.begin<cv::Vec<float, 3>>(); itr != linear.end<cv::Vec<float, 3>>();
        ++itr) {
            for (int channel = 0; channel < 3; ++channel) {
                (*itr)[channel] = srgb_to_linear((*itr)[channel]);
            }
        }
        return linear;
    }

// Converts an OpenCV Mat with linear values in the range [0, 1] to
// a 8-bit (CV_8UC3) OpenCV Mat in the range [0, 255].
    static cv::Mat linear_to_srgb(const cv::Mat& linear) {
        cv::Mat srgb(linear.rows, linear.cols, CV_8UC3);
        for (int r = 0; r < linear.rows; ++r) {
            for (int c = 0; c < linear.cols; ++c) {
                auto& dest = srgb.at<cv::Vec<unsigned char, 3>>(r, c);
                auto& source = linear.at<cv::Vec<float, 3>>(r, c);
                for (int channel = 0; channel < 3; ++channel) {
                    dest[channel] = cv::saturate_cast<unsigned char>(linear_to_srgb(source[channel]) + 0.5f);
                }
            }
        }
        return srgb;
    }
}

namespace ccn {
/**
 * @brief Checks if two images have the same size and type.
 *
 * @param[in] img1 The first image.
 * @param[in] img2 The second image.
 * @return True if the images have the same size and type, false otherwise.
 */
    bool CheckImageSizeAndType(const cv::Mat &img1, const cv::Mat &img2) {

        // Check if the images have the same size.
        if (img1.size() != img2.size()) {
            return false;
        }

        // Check if the images have the same type.
        if (img1.type() != img2.type()) {
            return false;
        }

        // The images have the same size and type.
        return true;
    }

/**
 * @brief Copies the lightness from one image to another.
 *
 * @param[in] newImg The new image.
 * @param[in] originImg The original image.
 * @return The output image, with the lightness of the original image copied to it.
 *
 * @throws std::invalid_argument If the images have different sizes or types.
 */
    cv::Mat CopyLightness(cv::Mat newImg, cv::Mat originImg) {

        // Check if the images have the same size.
        if (!CheckImageSizeAndType(newImg, originImg)) {
            throw std::invalid_argument("type1 != type2 or size1 != size2");
        }

        // Convert the images to linear space.
        newImg = vision::srgb_to_linear(newImg);
        originImg = vision::srgb_to_linear(originImg);

        // Create a new image to store the output.
        cv::Mat resultImg = newImg.clone();

        // For each pixel in the image...
        for (int x = 0; x < newImg.rows; x++) {
            for (int y = 0; y < newImg.cols; y++) {

                // Calculate the lightness of the new and original pixels.
                double lNew = newImg.at<cv::Vec3f>(x, y)[0] + newImg.at<cv::Vec3f>(x, y)[1] + newImg.at<cv::Vec3f>(x, y)[2];
                double lOld = originImg.at<cv::Vec3f>(x, y)[0] + originImg.at<cv::Vec3f>(x, y)[1] + originImg.at<cv::Vec3f>(x, y)[2];

                // Set the output pixel to the original lightness, scaled by the lightness of the new pixel.
                resultImg.at<cv::Vec3f>(x, y)[0] = newImg.at<cv::Vec3f>(x, y)[0] * (lOld / lNew);
                resultImg.at<cv::Vec3f>(x, y)[1] = newImg.at<cv::Vec3f>(x, y)[1] * (lOld / lNew);
                resultImg.at<cv::Vec3f>(x, y)[2] = newImg.at<cv::Vec3f>(x, y)[2] * (lOld / lNew);
            }
        }

        // Convert the output image back to sRGB space.
        resultImg = vision::linear_to_srgb(resultImg);

        // Return the output image.
        return resultImg;
    }

/**
 * @brief Calculates the average difference between two images.
 *
 * @param[in] img1 The first image.
 * @param[in] img2 The second image.
 * @return The average difference between the two images.
 *
 * @throws std::invalid_argument If the images have different sizes or types.
 */
    double CalculateAverageDiff(const cv::Mat &img1, const cv::Mat &img2) {

        // Check if the images have the same size.
        if (!CheckImageSizeAndType(img1, img2)) {
            throw std::invalid_argument("type1 != type2 or size1 != size2");
        }

        // Calculate the total difference between the two images.
        double totalDiff = 0.0;
        for (int i = 0; i < img1.rows; i++) {
            for (int j = 0; j < img1.cols; j++) {
                cv::Vec3b diff = img1.at<cv::Vec3b>(i, j) - img2.at<cv::Vec3b>(i, j);
                totalDiff += cv::norm(diff);
            }
        }

        // Calculate the average difference.
        double averageDiff = totalDiff / (img1.rows * img1.cols);

        // Return the average difference.
        return averageDiff;
    }

/**
 * @brief Converts an image to grayscale using the Gray World Assumption.
 *
 * The Gray World Assumption states that the average of all the pixels in an image
 * should be achromatic (gray). This function converts an image to grayscale by
 * dividing each channel by the average of the channels.
 *
 * @param[in,out] img The input image. The output image will be grayscale.
 *
 * @throws std::invalid_argument If the image is empty.
 */
    void GrayWorldAssumption(cv::Mat &img) {

        // Check if the image is empty.
        if (img.empty()) {
            throw std::invalid_argument("invalid size");
        }

        // Convert the image to linear space.
        img = vision::srgb_to_linear(img);

        // Split the image into channels.
        std::vector<cv::Mat> channels;
        cv::split(img, channels);

        // Calculate the average of each channel.
        double avgB = cv::mean(channels[0])[0];
        double avgG = cv::mean(channels[1])[0];
        double avgR = cv::mean(channels[2])[0];

        // Divide each channel by the average.
        channels[0] /= avgB * 3;
        channels[1] /= avgG * 3;
        channels[2] /= avgR * 3;

        // Merge the channels back into an image.
        cv::merge(channels, img);

        // Convert the image back to sRGB space.
        img = vision::linear_to_srgb(img);
    }

/**
 * @brief Normalizes the colors in an image.
 *
 * This function normalizes the colors in an image by dividing each channel by the sum of all the channels.
 *
 * @param[in,out] img The input image. The output image will have normalized colors.
 *
 * @throws std::invalid_argument If the image is empty.
 */
    void NormalizeColors(cv::Mat &img) {

        // Check if the image is empty.
        if (img.empty()) {
            throw std::invalid_argument("invalid size");
        }

        // Convert the image to linear space.
        img = vision::srgb_to_linear(img);

        // Split the image into channels.
        std::vector<cv::Mat> channels;
        cv::split(img, channels);

        // Calculate the sum of all the channels.
        cv::Mat sum = channels[0] + channels[1] + channels[2];

        // Divide each channel by the sum.
        channels[0] /= sum;
        channels[1] /= sum;
        channels[2] /= sum;

        // Merge the channels back into an image.
        cv::merge(channels, img);

        // Convert the image back to sRGB space.
        img = vision::linear_to_srgb(img);
    }



/**
* @brief Compresses the colors in an image using a normalized lightness approach.
*
* This function compresses the colors in an image by first normalizing the colors,
* then applying the Gray World Assumption.
*
* @param[in] img The input image.
* @param[in] convValue The learning rate. The learning rate controls how quickly the colors are compressed. A higher learning rate will result in more compression, while a lower learning rate will result in less compression.
* @param[in] CL Whether or not to apply the Gray World Assumption.
*
* @return The output image, with the colors compressed.
*
* @throws std::invalid_argument If the image is empty.
*/
    cv::Mat ComprColorImageNorm(const cv::Mat &img, const double &convValue, const bool &CL) {

        // Check if the image is empty.
        if (img.empty()) {
            throw std::invalid_argument("invalid size");
        }

        // Create a copy of the input image.
        cv::Mat resultImg = img.clone();

        // Create a temporary image to store the intermediate results.
        cv::Mat lastStepImg = resultImg;

        // Iterate until the learning rate is less than or equal to a threshold.
        for (double stepDiff = convValue; stepDiff >= convValue; stepDiff = CalculateAverageDiff(lastStepImg, resultImg)) {

            // Normalize the colors in the image.
            NormalizeColors(resultImg);

            // Apply the Gray World Assumption to the image.
            GrayWorldAssumption(resultImg);

            // Copy the current image to the temporary image.
            lastStepImg = resultImg;
        }

        // If the `CL` flag is set, apply the Gray World Assumption to the image.
        if (CL) {
            resultImg = CopyLightness(resultImg,img);
        }

        // Return the output image.
        return resultImg;
    }

/**
* @brief Calculates the difference between two images.
*
* This function calculates the difference between two images by computing the absolute difference between each pixel in the images.
*
* @param[in] img1 The first image.
* @param[in] img2 The second image.
*
* @return The output image, with the difference between the two images.
*
* @throws std::invalid_argument If the images have different sizes or types.
*/
    cv::Mat CalculateDiff(const cv::Mat &img1, const cv::Mat &img2) {

        // Check if the images have the same size.
        if (!CheckImageSizeAndType(img1, img2)) {
            throw std::invalid_argument("type1 != type2 or size1 != size2");
        }

        // Create a new image to store the difference.
        cv::Mat dif;

        // Calculate the absolute difference between the two images.
        cv::absdiff(img1, img2, dif);

        // Return the output image.
        return dif;
    }


/**
* @brief Calculates the difference between two images using a normalized lightness approach.
*
* This function compresses the colors in an image by first normalizing the colors,
* then applying the Gray World Assumption. The function does not take a custom lightness function as input.
*
* The function also returns a vector of images, where each image in the vector
* represents the difference between the current image and the previous image.
* This vector can be used to visualize the progress of the compression algorithm.
*
* @param[in] img The input image.
* @param[in] convValue The learning rate. The learning rate controls how quickly the colors are compressed. A higher learning rate will result in more compression, while a lower learning rate will result in less compression.
*
* @return A vector of images, where each image in the vector represents the difference between the current image and the previous image.
*
* @throws std::invalid_argument If the image is empty.
*/
    std::vector<cv::Mat> ComprColorImageNormDiff(const cv::Mat &img, const double &lr) {
        // Check if the image is empty.
        if (img.empty()) {
            throw std::invalid_argument("invalid size");
        }

        // Create a copy of the input image.
        cv::Mat resultImg = img.clone();

        // Create a temporary image to store the intermediate results.
        cv::Mat lastStepImg = resultImg;

        // Create a vector to store the differences.
        std::vector<cv::Mat> diff;

        // Iterate until the learning rate is less than or equal to a threshold.
        for (double stepDiff = lr; stepDiff >= lr; stepDiff = CalculateAverageDiff(lastStepImg, resultImg)) {

            // Copy the current image to the temporary image.
            lastStepImg = resultImg;

            // Normalize the colors in the image.
            NormalizeColors(resultImg);

            // Apply the Gray World Assumption to the image.
            GrayWorldAssumption(resultImg);

            // Calculate the difference between the current image and the previous image.
            // Add the difference to the vector of differences.
            diff.push_back(CalculateDiff(CopyLightness(lastStepImg, img), CopyLightness(resultImg, img)));
        }
        // Return the vector of differences.
        return diff;
    }
}