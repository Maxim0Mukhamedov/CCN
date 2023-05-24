#include "ccn.hpp"
#include <iostream>
cv::Mat mergeImage(cv::Mat img1, cv::Mat img2)
{
    int rows = (img1.rows > img2.rows) ? img1.rows : img2.rows;
    int cols = img1.cols + img2.cols;
    cv::Mat3b res(rows, cols, cv::Vec3b(0,0,0));
    img1.copyTo(res(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(res(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
    return res;
}
int main() {
    // Load the input 3RGB image.
    cv::Mat image = cv::imread("/home/maxim/CLionProjects/CurseJob/test_input/image1.jpg");
    cv::imwrite("/home/maxim/CLionProjects/CurseJob/test_output/normalized_image.jpeg",
                mergeImage(image,
                           ComprColorImageNorm(image.clone())));
    return 0;

}



