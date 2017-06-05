#include <iostream>
#include <unistd.h>
#include "opencv2/core.hpp"
//#include "opencv2/viz.hpp"
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
/*
 * Load image. done
 * Show image.  done
 * Add matrices and function. done 
 * Create function that transforms all pixel of image and creates a new iamge. done
 * Show the new image in separate window. done
 * Fix function until image is ok.
 *
 */
#define IMAGE "../img1.jpg"

float f(float r) {
    if (r == 0) return 1; 
    return  atan(2 * r * tan(0.926464 / 2)) / (0.926464*r);
}

Mat_<int> getNewCoordinates(int row, int col, Mat KUnproject, Mat KProject) {
    cv::Mat_<float> input(3/*rows*/,1 /*cols */); 
    input(0,0) = col;
    input(1,0) = row;
    input(2,0) = 1;
    input = KUnproject*input;

    input = input/input(2,0);
   
    float r = sqrt(input(0,0)*input(0,0)+input(1,0)*input(1,0));

    input = f(r)*input;
    input(2,0) = 1; //unprojected, now we set the distance to some constant, because we do not know it
    Mat newCoordinates = KProject*input; //project back
    return newCoordinates;      
}

int main()
{
    cout << "Built with OpenCV " << CV_VERSION << endl;
    Mat_<uchar> image;
    image = imread(IMAGE, IMREAD_GRAYSCALE);   
    if(!image.data) {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    
    imshow( "Display window", image);

    float k[] = {388.6, 0, 343.7,
               0, 389.4, 234.6,
               0, 0, 1};
    cv::Mat K = Mat(3, 3, CV_32F, k);

    float ku[] = {250, 0, 512,
               0, 250, 384,
               0, 0, 1};
    cv::Mat Ku = Mat(3, 3, CV_32F, ku);

    cv::Mat_<uchar> newImage(768, 1024);   
    cv::Mat_<int> newCoord;

    int newRow=0, newCol=0;

    for (int  row = 0; row < newImage.rows; row++) {
        for(int col = 0; col < newImage.cols; col++) {
            newCoord = getNewCoordinates(row, col, Ku.inv(), K);
            newRow = (int) round(newCoord(1, 0));
            newCol = (int) round(newCoord(0, 0));
            
            if (newRow < image.rows && newCol < image.cols && newRow > 0 && newCol > 0) { 
                newImage(row, col) = image(newRow, newCol);
            }

        }
    }
    imshow("Display window 2", newImage);

    waitKey(0);
    return 0;
}
