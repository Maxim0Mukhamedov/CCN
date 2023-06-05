#include "ccn.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "string"
#include "iostream"
cv::Mat mergeImage(cv::Mat img1, cv::Mat img2)
{
    int rows = (img1.rows > img2.rows) ? img1.rows : img2.rows;
    int cols = img1.cols + img2.cols;
    cv::Mat3b res(rows, cols, cv::Vec3b(0,0,0));
    img1.copyTo(res(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(res(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
    return res;
}
// save display save-directory input-directory name of image(from input) conv-value CopyLightness conv-progress
int main(int argc, char **argv ) {
    // Load the input 3RGB image.
    bool save = false;
    bool show = false;
    bool copyLightness = false;
    bool convProgress = false;
    double conValue = 0.1;
    std::string saveDirectory = INSTALL_PATH + static_cast<std::string>("/test_output");
    std::string inputDirectory = INSTALL_PATH + static_cast<std::string>("/test_input/test_image.jpg");
    for (int i = 1; i < argc; i ++) {
        if (static_cast<std::string>(argv[i]) == "-S") {
            save = true;
        } else if (static_cast<std::string>(argv[i]) == "-D") {
            show = true;
        } else if (static_cast<std::string>(argv[i]) == "-L") {
            copyLightness = true;
        } else if (static_cast<std::string>(argv[i]) == "-P") {
            convProgress = true;
        } else if (static_cast<std::string>(argv[i]) == "-h" || static_cast<std::string>(argv[i]) == "-help"){
            std::cout << "THE PARAMS OF CNN\n";
            std::cout << "-S : save the results to output directory" << '\n';
            std::cout << "-D : show the results" << '\n';
            std::cout << "-L : use CopyLightness alg" << '\n';
            std::cout << "-P : save the converges progression" << '\n';
            std::cout << "next params should use on this way: -[param]=[option]" << '\n';
            std::cout << "-v : change the converges value" << '\n';
            std::cout << "-i : change input directory" << '\n';
            std::cout << "-s : change output directory and save results there" << '\n';
            std::cout << "-h (-help): show params meaning" << std::endl;
            return 0;
        } else if (static_cast<std::string>(argv[i]).size() > 3) {
            std::string param = static_cast<std::string>(argv[i]);
            if (param[0] == '-' && param[2] == '=') {
                if (param[1] == 'v') {
                    conValue = std::stod(param.substr(3,param.size() - 3));
                } else if (param[1] == 'i') {
                    inputDirectory = param.substr(3,param.size() - 3);
                } else if (param[1] == 's') {
                    saveDirectory = param.substr(3,param.size() - 3);
                    save = true;
                } else {
                    throw std::invalid_argument("invalid argument");
                }
            } else {
                throw std::invalid_argument("invalid argument");
            }
        } else {
            throw std::invalid_argument("invalid argument");
        }
    }
    cv::Mat image = cv::imread(inputDirectory);
    if (image.empty()) {
        std::cout << "Could not read the image." << std::endl;
        return 1;
    }
    cv::Mat diffprogression;
    if (convProgress) {
        std::vector<cv::Mat> HMC = ccn::ComprColorImageNormDiff(image.clone());
        diffprogression = HMC[0];
        for (int i = 1; i < HMC.size(); i++) {
            HMC[i].convertTo(HMC[i], CV_8UC3, 128);
            diffprogression = mergeImage(diffprogression, HMC[i]);
        }
        if (save) {
            cv::imwrite(saveDirectory + "/diff.jpg",
                        mergeImage(image,
                                   diffprogression));
        }
        std::cout << HMC.size();
    }
    cv::Mat result = ccn::ComprColorImageNorm(image.clone(), conValue, copyLightness);
    if (save) {
        cv::imwrite(saveDirectory + "/normalized_image.jpg",
                    mergeImage(image,result));
    }
    if (show) {
        // Display the image.
        namedWindow("CCN result", cv::WINDOW_AUTOSIZE);
        imshow("CCN result", mergeImage(image,result));
        if (convProgress) {
            namedWindow("CCN progression", cv::WINDOW_AUTOSIZE);
            imshow("CCN progression", diffprogression);
        }
        // Wait for a keystroke in the window.
        int k = cv::waitKey(0);
        // Close the window.
        cv::destroyAllWindows();
    }
    return 0;

}



