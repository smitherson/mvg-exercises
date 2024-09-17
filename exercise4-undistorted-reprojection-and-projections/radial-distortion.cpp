#include <fstream>
#include <iostream>

#include "opencv2/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <unistd.h>

using namespace cv;
using namespace std;

#define IMAGE "/home/smith/Datasets/calibration/maxi-car/20240913165608.jpg"
//#define CALIB_FILE "/home/smith/Datasets/calibration/maxi-car/calib-maxi-car.json"
#define CALIB_FILE "/home/smith/Datasets/calibration/maxi-car/calib-maxi-car-no-dist.json"

#include "json.hpp"
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Function to manually undistort fisheye image and reproject with pinhole camera model
void fisheyeToPinhole(const Mat &fisheyeImage, Mat &pinholeImage,
                      const Mat &K_fisheye, const Mat &D_fisheye,
                      const Mat &K_pinhole, const float mirrorOffset, const Size &pinholeSize) {

    pinholeImage = Mat::zeros(pinholeSize, fisheyeImage.type());

    cout << "pinholeImage rows, cols: " << pinholeImage.rows << " " << pinholeImage.cols << std::endl;
    cout << "fisheyeImage rows, cols: " << fisheyeImage.rows << " " << fisheyeImage.cols << std::endl;

    float k1 = D_fisheye.at<float>(0, 0);
    float k2 = D_fisheye.at<float>(0, 1);
    // float k3 = D_fisheye.at<float>(0, 2);
    // float k4 = D_fisheye.at<float>(0, 3);

    float fx_fish = K_fisheye.at<float>(0, 0);
    float fy_fish = K_fisheye.at<float>(1, 1);
    float cx_fish = K_fisheye.at<float>(0, 2);
    float cy_fish = K_fisheye.at<float>(1, 2);

    float fx_pin = K_pinhole.at<float>(0, 0);
    float fy_pin = K_pinhole.at<float>(1, 1);
    float cx_pin = K_pinhole.at<float>(0, 2);
    float cy_pin = K_pinhole.at<float>(1, 2);

    std::cout << "pinholeSize.height " << pinholeSize.height << "\n";

    for (int piholeRow = 0; piholeRow < pinholeImage.rows; piholeRow++) {
        for (int pinholeCol = 0; pinholeCol < pinholeImage.cols; pinholeCol++) {

            /*double x_norm = (pinholeCol - cx_pin) / fx_pin;
            double y_norm = (piholeRow - cy_pin) / fy_pin;
            double r_norm = sqrt(x_norm * x_norm + y_norm * y_norm);

            // Apply inverse of radial distortion to map back to the fisheye image
            double theta = atan(r_norm);
            double theta_d = theta * (1 + k1 * pow(theta, 2) + k2 * pow(theta, 4)) ;//+ k3 * pow(theta, 6) + k4 * pow(theta, 8));

            // Convert back to distorted (fisheye) coordinates
            int x_fish = cvRound(fx_fish * x_norm * theta_d / r_norm + cx_fish);
            int y_fish = cvRound(fy_fish * y_norm * theta_d / r_norm + cy_fish);

            // Check if the coordinates fall within the fisheye image bounds
            if (x_fish >= 0 && x_fish < fisheyeImage.cols && y_fish >= 0 && y_fish < fisheyeImage.rows) {
                //printf("%d %d max %d %d\n", x_fish, y_fish, fisheyeImage.cols, fisheyeImage.rows);
                // Map the fisheye image pixel to the pinhole image
                pinholeImage.at<Vec3b>(piholeRow, pinholeCol) = fisheyeImage.at<Vec3b>(cvRound(y_fish), cvRound(x_fish));
            }*/

            float x_tau = (pinholeCol - cx_pin) / fx_pin;
            float y_tau = (piholeRow - cy_pin) / fy_pin;

            float coeff = (mirrorOffset + sqrt(1 + (1 - mirrorOffset * mirrorOffset) * (x_tau * x_tau + y_tau + y_tau))) /
                          (x_tau * x_tau + y_tau + y_tau + 1);

            float x_unproj = coeff * x_tau / (coeff - mirrorOffset);
            float y_unproj = coeff * y_tau / (coeff - mirrorOffset);

            int x_fish = cvRound(fx_fish * x_unproj + cx_fish);
            int y_fish = cvRound(fy_fish * y_unproj + cy_fish);
            if (x_fish >= 0 && x_fish < fisheyeImage.cols && y_fish >= 0 && y_fish < fisheyeImage.rows) {
                pinholeImage.at<Vec3b>(piholeRow, pinholeCol) = fisheyeImage.at<Vec3b>(y_fish, x_fish);
            }
        }
    }
}

int main() {
    // Load the fisheye image
    Mat fisheyeImage = imread(IMAGE);
    if (fisheyeImage.empty()) {
        cout << "Error: Could not load the fisheye image!" << endl;
        return -1;
    }

    std::ifstream calibrationFileStream(CALIB_FILE);

    nlohmann::json calibrationData = nlohmann::json::parse(calibrationFileStream);

    int calibrationRows = calibrationData["resolution"]["y"];
    int calibrationCols = calibrationData["resolution"]["x"];

    assert(calibrationCols == fisheyeImage.cols && calibrationRows == fisheyeImage.rows);

    float fx = calibrationData["focal_length"]["x"]; // 700.1441001496614 / rows;
    float fy = calibrationData["focal_length"]["y"]; // 713.1370873930887 / cols;
    float cx = calibrationData["center_pixel"]["x"]; // 321.5265928650794 / rows;
    float cy = calibrationData["center_pixel"]["y"]; // 240.1270406206272 / cols;

    float r0 = calibrationData["radial_distortion"]["r0"];
    float r1 = calibrationData["radial_distortion"]["r1"];

    float mirrorOffset = calibrationData["mirrorOffset"];

    Mat K_fisheye = (Mat_<float>(3, 3) << fx, 0.0, cx,
                     0.0, fy, cy,
                     0.0, 0.0, 1.0);

    Mat D_fisheye = (Mat_<float>(1, 2) << r0, r1);

    Size pinholeSize(fisheyeImage.cols * 1.5, fisheyeImage.rows * 1.5);

    Mat K_pinhole = (Mat_<float>(3, 3) << 200.0, 0.0, pinholeSize.width / 2,
                     0.0, 200.0, pinholeSize.height / 2,
                     0.0, 0.0, 1.0);

    Mat pinholeImage;
    fisheyeToPinhole(fisheyeImage, pinholeImage, K_fisheye, D_fisheye, K_pinhole, mirrorOffset, pinholeSize);

    // Save or display the resulting pinhole projection image
    imshow("Fisheye Projection", fisheyeImage);
    imshow("Pinhole Projection", pinholeImage);

    waitKey(0);

    return 0;
}
