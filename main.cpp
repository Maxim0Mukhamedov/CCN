#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

/**
 * @brief Copies the lightness of one image to another.
 *
 * This function copies the lightness of the input image to the output image.
 * The lightness is calculated as the weighted sum of the red, green, and blue channels.
 * The weights are specified by the `wr`, `wg`, and `wb` parameters.
 *
 * @param image The input image.
 * @param InputImage The image to copy the lightness to.
 * @param wr The weight for the red channel.
 * @param wg The weight for the green channel.
 * @param wb The weight for the blue channel.
 * @return The output image.
 */
cv::Mat CopyLightness(cv::Mat image, cv::Mat InputImage, float wr = 1.0f, float wg = 1.0f, float wb = 1.0f) {
    // Convert the input and output images to the floating-point format `CV_32FC3`.
    image.convertTo(image, CV_32FC3);
    InputImage.convertTo(InputImage, CV_32FC3);

    // Create a clone of the input image to store the output image.
    cv::Mat OutputImage = image.clone();

    // Iterate over all the pixels in the image.
    for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {
            // Calculate the lightness of the input image at the current pixel.
            float Lold = wr * image.at<cv::Vec3f>(x, y)[0] + wg * image.at<cv::Vec3f>(x, y)[1] + wb * image.at<cv::Vec3f>(x, y)[2];

            // Calculate the lightness of the output image at the current pixel.
            float Lnew = wr * InputImage.at<cv::Vec3f>(x, y)[0] + wg * InputImage.at<cv::Vec3f>(x, y)[1] + wb * InputImage.at<cv::Vec3f>(x, y)[2];

            // Set the output image pixel to the lightness of the input image, scaled by the lightness of the output image.
            OutputImage.at<cv::Vec3f>(x, y)[0] = image.at<cv::Vec3f>(x, y)[0] * (Lnew / Lold);
            OutputImage.at<cv::Vec3f>(x, y)[1] = image.at<cv::Vec3f>(x, y)[1] * (Lnew / Lold);
            OutputImage.at<cv::Vec3f>(x, y)[2] = image.at<cv::Vec3f>(x, y)[2] * (Lnew / Lold);
        }
    }

    // Convert the output image back to the unsigned integer format `CV_8UC3`.
    image.convertTo(image, CV_8UC3);
    InputImage.convertTo(InputImage, CV_8UC3);

    // Return the output image.
    return OutputImage;
}

/**
 * @brief Calculates the average value of a channel in an image.
 *
 * This function calculates the average value of a channel in an image.
 * The channel is specified by the `channel` parameter.
 *
 * @param image The image.
 * @param channel The channel to calculate the average of.
 * @return The average value of the channel.
 */
float AverageChannelValue(const cv::Mat& image, const int& channel) {
    // Create a variable to store the total number of channel values.
    float total_channel_values = 0;
    // Create a variable to store the average number of channel values.
    float average_channel_value = 0.0f;
    // Iterate over each pixel in the image.
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            // Add the channel value to the total number of red channel values.
            total_channel_values += image.at<cv::Vec3f>(i, j)[channel];
        }
    }
    // Calculate the average number of channel values.
    average_channel_value = total_channel_values / (image.rows * image.cols);
    return average_channel_value;
}

/**
 * @brief Calculates the average difference between the pixels of two images.
 *
 * @param image1 The first image.
 * @param image2 The second image.
 * @return The average difference between the pixels of the two images.
 */
float CalculateAverageDiff(cv::Mat image1, cv::Mat image2) {

    // Check if the images have the same size.
    // If they do not, return 0.0.
    if (image1.size() != image2.size()) {
        return 0.0;
    }

    // Calculate the total difference between the pixels.
    float totalDiff = 0.0;

    // Iterate over each pixel in the images.
    for (int i = 0; i < image1.rows; i++) {
        for (int j = 0; j < image1.cols; j++) {

            // Calculate the difference between the pixels.
            cv::Vec3b diff = image1.at<cv::Vec3b>(i, j) - image2.at<cv::Vec3b>(i, j);

            // Add the difference to the total difference.
            totalDiff += cv::norm(diff);
        }
    }

    // Calculate the average difference.
    float averageDiff = totalDiff / (image1.rows * image1.cols);

    // Return the average difference.
    return averageDiff;
}

/**
 * @brief Corrects an image for color constancy using the gray world assumption.
 *
 * This function corrects an image for color constancy using the gray world assumption.
 * The gray world assumption states that the average of all the pixels in an image should be gray. 
 * This function calculates the average of each channel in the image and then divides each channel by its average.
 *
 * @param image The image to correct.
 * @return The corrected image.
 */
cv::Mat GrayWorldAssumption(cv::Mat image) {
    // Convert the image to the floating-point format.
    image.convertTo(image, CV_32FC3);

    // Calculate the average of each channel in the image.
    float avgR = AverageChannelValue(image, 0) * 3;
    float avgG = AverageChannelValue(image, 1) * 3;
    float avgB = AverageChannelValue(image, 2) * 3;

    // Create a array to store the scale values for each channel.
    float scaleValue[3] = {avgR, avgG, avgB};

    // Iterate over each channel in the image.
    for (int channel = 0; channel < 3; ++channel) {
        // Iterate over each pixel in the channel.
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                // Divide the pixel value by the scale value for the channel.
                image.at<cv::Vec3f>(i, j)[channel] = image.at<cv::Vec3f>(i, j)[channel] / scaleValue[channel];
            }
        }
    }

    // Convert the image back to the unsigned integer format.
    image.convertTo(image, CV_8UC3, 255.);

    // Return the corrected image.
    return image;
}

/**
 * @brief Normalizes the colors in an image.
 *
 * This function normalizes the colors in an image by dividing each pixel by the sum of its RGB values. This ensures that all pixels in the image have a value between 0 and 1.
 *
 * @param image The image to normalize.
 * @return The normalized image.
 */
cv::Mat NormalizeColors(cv::Mat image) {
    // Convert the image to the floating-point format.
    image.convertTo(image, CV_32FC3);

    // Iterate over each pixel in the image.
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            // Calculate the sum of the RGB values at the current pixel.
            float Sum = image.at<cv::Vec3f>(i, j)[0] + image.at<cv::Vec3f>(i, j)[1] + image.at<cv::Vec3f>(i, j)[2];

            // If the sum is greater than 0, divide each RGB value by the sum.
            if (Sum > 0) {
                image.at<cv::Vec3f>(i, j)[0] = image.at<cv::Vec3f>(i, j)[0] / Sum;
                image.at<cv::Vec3f>(i, j)[1] = image.at<cv::Vec3f>(i, j)[1] / Sum;
                image.at<cv::Vec3f>(i, j)[2] = image.at<cv::Vec3f>(i, j)[2] / Sum;
            }
        }
    }

    // Convert the image back to the unsigned integer format.
    image.convertTo(image, CV_8UC3, 255.);

    // Return the normalized image.
    return image;
}

/**
 * @brief Performs color compression on an image.
 *
 * This function performs color compression on an image by applying the gray world assumption and then normalizing the colors. The gray world assumption states that the average of all the pixels in an image should be gray. This function calculates the average of each channel in the image and then divides each channel by its average. The colors are then normalized by dividing each pixel by the sum of its RGB values.
 *
 * @param image The image to compress.
 * @param CL Whether to copy the lightness of the original image to the compressed image.
 * @param lr The convergence value.
 * @return The compressed image.
 */
cv::Mat ComprColorImageNorm(cv::Mat image, bool CL = true, float lr = 0.1) {
    // Create a clone of the input image to store the compressed image.
    cv::Mat result = image.clone();
    cv::Mat last_step = result;
    // Iterate until convergence.
    for (float step_diff = lr; step_diff >= lr; step_diff = CalculateAverageDiff(last_step,result)) {
        // Save the last step of calculates.
        last_step = result;
        // Normalize the colors in the image.
        result = NormalizeColors(result);
        // Apply the gray world assumption to the image.
        result = GrayWorldAssumption(result);
    }

    // If the `CL` flag is set, copy the lightness of the original image to the compressed image.
    if (CL) {
        result = CopyLightness(result, image, 1, 1, 1);
    }

    // Return the compressed image.
    return result;
}

int main() {
    // Load the input image.
    cv::Mat image = cv::imread("/home/maxim/CLionProjects/CurseJob/test_input/fig3.jpg");
    cv::imwrite("/home/maxim/CLionProjects/CurseJob/test_output/normalized_image.jpeg", ComprColorImageNorm(image.clone()));
    return 0;

}


