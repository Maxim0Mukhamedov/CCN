#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "srgb.h"

cv::Mat CopyLightness(cv::Mat newImg, cv::Mat originImg) {
    float wr=0.21; float wg=0.7; float wb=0.07;
    newImg = vision::srgb_to_linear(newImg);
    originImg = vision::srgb_to_linear(originImg);
    cv::Mat resultImg = newImg.clone();
    for (int x = 0; x < newImg.rows; x++) {
        for (int y = 0; y < newImg.cols; y++) {
            float lNew = wb * newImg.at<cv::Vec3f>(x, y)[0]
                    + wg * newImg.at<cv::Vec3f>(x, y)[1]
                    + wr * newImg.at<cv::Vec3f>(x, y)[2];
            float lOld = wb * originImg.at<cv::Vec3f>(x, y)[0]
                    + wg * originImg.at<cv::Vec3f>(x, y)[1]
                    + wr * originImg.at<cv::Vec3f>(x, y)[2];
            resultImg.at<cv::Vec3f>(x, y)[0] = newImg.at<cv::Vec3f>(x, y)[0] * (lOld / lNew);
            resultImg.at<cv::Vec3f>(x, y)[1] = newImg.at<cv::Vec3f>(x, y)[1] * (lOld / lNew);
            resultImg.at<cv::Vec3f>(x, y)[2] = newImg.at<cv::Vec3f>(x, y)[2] * (lOld / lNew);
        }
    }
    return resultImg;
}

float CalculateAverageDiff(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat dif = img1 - img2;
    float averageDiff = (cv::sum(dif)[0] + cv::sum(dif)[1] + cv::sum(dif)[2])  / (img1.rows * img1.cols);
    return averageDiff;
}
void GrayWorldAssumption(cv::Mat& img) {
    std::vector<cv::Mat> channels;
    cv::split(img,channels);
    float avgB = cv::mean(channels[0])[0];
    float avgG = cv::mean(channels[1])[0];
    float avgR = cv::mean(channels[2])[0];
    channels[0] /= avgB * 3;
    channels[1] /= avgG * 3;
    channels[2] /= avgR * 3;
    cv::merge(channels,img);
}
void NormalizeColors(cv::Mat& img) {
    std::vector<cv::Mat> channels;
    cv::split(img,channels);
    cv::Mat sum = channels[0] + channels[1] + channels[2];
    channels[0] /= sum;
    channels[1] /= sum;
    channels[2] /= sum;
    cv::merge(channels,img);
}
cv::Mat ComprColorImageNorm(const cv::Mat& img, const bool& CL = true, const float& lr = 0.1, const std::function<cv::Mat(cv::Mat,cv::Mat)>& CLF = CopyLightness) {
    cv::Mat resultImg = vision::srgb_to_linear(img.clone());
    cv::Mat lastStepImg = resultImg;

    for (float stepDiff = lr; stepDiff>= lr; stepDiff = CalculateAverageDiff(lastStepImg,resultImg)) {
        lastStepImg = resultImg;
        NormalizeColors(resultImg);
        GrayWorldAssumption(resultImg);
    }
    if (CL) {
        resultImg = CopyLightness(resultImg, img);
    }
    return vision::linear_to_srgb(resultImg);
}

//cv::Mat mergeImage(cv::Mat img1, cv::Mat img2)
//{
//    int rows = (img1.rows > img2.rows) ? img1.rows : img2.rows;
//    int cols = img1.cols + img2.cols;
//    cv::Mat3b res(rows, cols, cv::Vec3b(0,0,0));
//    img1.copyTo(res(cv::Rect(0, 0, img1.cols, img1.rows)));
//    img2.copyTo(res(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
//    return res;
//}