#include <vector>
#include <iostream>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/viz.hpp"
#include <limits>       // std::numeric_limits
#include <unistd.h>
#include "../common.h"

using namespace std;
using namespace cv;
#define k_values { 535.4, 0, 320.1, 0, 539.2, 247.6, 0, 0, 1}

#define IMAGE1 "./batinria0.pgm"
#define IMAGE2 "./batinria1.pgm"

typedef float MatType;


int main() {

Mat img1 = imread(IMAGE1, IMREAD_GRAYSCALE);
Mat img2 = imread(IMAGE2, IMREAD_GRAYSCALE);

//% Left camera parameters:
MatType k1[] = {844.310547, 0, 243.413315, 0, 1202.508301, 281.529236, 0, 0, 1};

MatType g1_values[] = {0.655133, 0.031153, 0.754871, -793.848328,
               0.003613, 0.999009, -0.044364, 269.264465,
              -0.755505, 0.031792, 0.654371, -744.572876,
               0       , 0       , 0       , 1};


//% Right camera parameters:
MatType k2[] = {852.721008, 0, 252.021805, 0, 1215.657349, 288.587189, 0, 0, 1};
MatType g2_values[] = {0.739514, 0.034059, 0.672279, -631.052917,
              -0.006453, 0.999032, -0.043515, 270.192749,
              -0.673111, 0.027841, 0.739017, -935.050842,
               0       , 0       , 0       , 1};

//% Compute the fundamental matrix:
Mat_<MatType> g1 = Mat_<MatType>(4,4, g1_values);
Mat_<MatType> g2 = Mat_<MatType>(4,4, g2_values);
Mat_<MatType> K1 = Mat_<MatType>(3,3, k1);
Mat_<MatType> K2 = Mat_<MatType>(3,3,  k2);


Mat_<MatType> g = (g2.inv()) * (g1);

Mat t = g(Rect(3,0,1,3));
Mat R = g(Rect(0,0,3,3));

//cout << "g: "; printMat(g);

//cout << "T: "; printMat(T);
//cout << "hat T: "; printMat(hat<MatType>(T));
//cout << "R: "; printMat(R);
//cout << "transpose(K2.inv()): "; printMat(transpose(K2.inv()));
//cout << "hat<MatType>(T): "; printMat(hat<MatType>(T));
//cout << "R: "; printMat(R);
//cout << "K1.inv(): "; printMat( K1.inv());

cout << "K2.inv: "; printMat(K2.inv());

Mat_<MatType> F = transpose(K2.inv()) * hat<MatType>(t) * R * K1.inv();
cout << "F: "; printMat(F);

int x, y;
x = 68;
y = 90;

//plot(x1,y1,'r+');
//hold off;

//% Compute epipolar line for x1:
Mat_<MatType> point = Mat_<MatType>(3,1);
point(0,0) = x;
point(0,1) = y;
point(0,2) = 1;

printMat(point);
Mat_<MatType> l = F * point;

//printMat(F);
cout<< "l: ";printMat(l);
int w = img2.cols;
//% Draw epipolar line in image2:
//figure; imshow(uint8(image2));
//hold on;
MatType m =  -l(0,0)/l(1,0);
MatType b = -l(2,0)/l(1,0);
cout << "m: " << m << " b: " << b << endl;

int y1 = (int) round( m * 1 + b);
int y2 = (int) round(m * w + b);

//line([1 w],[y1 y2])
line(img2, Point(1,y1), Point(w,y2), Scalar(255,255,255)); 
circle(img1, Point(x,y), 4, Scalar(255,255,255)); 

cout << w <<", "<< y1 << ", " << y2 << endl;

imshow("image1", img1);
imshow("image2", img2);
waitKey(0);
return 0;
}

