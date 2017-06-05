#include <iostream>
#include <unistd.h>
#include "opencv2/core.hpp"
#include "opencv2/viz.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
/*
 * 
 */
#define IMAGE1 "./image.png"
#define IMAGE2 "./image2.png"
#define SIGMA 3

float gaus(int x, int y) {
    float  sigma = SIGMA;
    float w = exp(-sqrt(x*x + y*y) / (2.0 * sigma * sigma)) / (2.0 * M_PI * sigma * sigma);
    ///cout << "x,y=" << x << ","<<y<<" "<<w << endl;
    return w;
}

int window = 2*SIGMA+1;
Mat getM(int x, int y, Mat Ix, Mat Iy, Mat Ixy) {
    
    Mat M = Mat::zeros(2,2, CV_32FC1);
    for(int i = x-SIGMA; i <x+SIGMA; i++) {
        for(int j = y-SIGMA; j < y+SIGMA; j++) { 
            M.at<float>(0,0) += gaus(i-x,j-y)*(Ix.at<char>(i,j)*Ix.at<char>(i,j)) ;
            M.at<float>(0,1) += gaus(i-x,j-y)*(Ix.at<char>(i,j)*Iy.at<char>(i,j)) ;
            M.at<float>(1,1) += gaus(i-x,j-y)*(Iy.at<char>(i,j)*Iy.at<char>(i,j)) ;
            M.at<float>(1,0) += gaus(i-x,j-y)*(Ix.at<char>(i,j)*Iy.at<char>(i,j)) ;
        }
    }
    //cout<< "M: " << M << endl;
    return M;
}
Mat getq(int x, int y, Mat Ix, Mat Iy, Mat dt) {
    
    Mat q = Mat::zeros(2,1, CV_32FC1);
    for(int i = x-SIGMA; i < x+SIGMA; i++) {
        for(int j = y-SIGMA; j < y+SIGMA; j++) { 
            //cout << "dt: " << (int) dt.at<char>(i,j) << endl;
            q.at<float>(0,0) += gaus(i-x,j-y)*(Ix.at<char>(i,j)*dt.at<char>(i,j)) ;
            q.at<float>(0,1) += gaus(i-x,j-y)*(Iy.at<char>(i,j)*dt.at<char>(i,j)) ;
        }
    }
    //cout<< "M: " << M << endl;
    return q;
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main() 
{
    cout << "Built with OpenCV " << CV_VERSION << endl;

    Mat image = imread(IMAGE1);
    Mat image2 = imread(IMAGE2);

    cvtColor( image, image, CV_BGR2GRAY);
    cvtColor( image2, image2, CV_BGR2GRAY);

    string ty =  type2str( image.type() );
    printf("Matrix: %s %dx%d \n", ty.c_str(), image.cols, image.rows );
    Mat Ix, Iy, Ixy;
    calculateGradient(image, Ix, Iy, Ixy);

    Mat Score = Mat(image.rows, image.cols, CV_8UC1);

    Mat dt = image - image2;
    for (int i = SIGMA; i < image.rows-SIGMA;i++) {
        for(int j = SIGMA; j < image.cols-SIGMA;j++) {
                Mat M = getM(i,j, Ix,Iy, Ixy);
                Score.at<char>(i,j) = (char)( determinant(M)+0.05*(M.at<float>(0,0)+M.at<float>(1,1) ));
                if (Score.at<char>(i,j) > 20 && Score.at<char>(i,j) > Score.at<char>(i-1,j-1) &&
                        Score.at<char>(i,j) > Score.at<char>(i-1,j) &&
                        Score.at<char>(i,j) > Score.at<char>(i-1,j+1) &&
                        Score.at<char>(i,j) > Score.at<char>(i,j+1) &&
                        Score.at<char>(i,j) > Score.at<char>(i+1,j+1) &&
                        Score.at<char>(i,j) > Score.at<char>(i+1,j) &&
                        Score.at<char>(i,j) > Score.at<char>(i+1,j-1) &&
                        Score.at<char>(i,j) > Score.at<char>(i,j-1) ) {
                    //image.at<char>(i,j) = 127;
                    Mat q = getq(i, j, Ix, Iy, dt);
                    //cout << "q: " << q << endl;
                    Mat vel = M.inv()*q;
                    //cout << vel << endl << endl;
                                       
                    char vx =(char) vel.at<float>(0,0);
                    char vy =(char) vel.at<float>(1,0);
                    if ( vx*vx+vy*vy >= 1 ) {
                        //cout <<(int) vx << ", " << (char) vy << endl; 
                        //image2.at<char>(vy+i,vx+j) = 127;
                        line(image, Point(j,i), Point(j,i), Scalar(255,255,255));
                        line(image2, Point(j+vx,i+vy), Point(j+vx,i+vy), Scalar(255,255,255));

                        //image2.at<char>(i+vy,j+vx) = 27;

                    }

                }
                
        }
    }
//    namedWindow("Sxy", WINDOW_AUTOSIZE );
//    imshow( "Sxy", Score);
    namedWindow("Original image", WINDOW_AUTOSIZE );
    imshow( "Original image", image);
    namedWindow("Original image 2", WINDOW_AUTOSIZE );
    imshow( "Original image 2", image2);

    namedWindow("Dt", WINDOW_AUTOSIZE );
    imshow( "Dt", dt);

    namedWindow("Gx", WINDOW_AUTOSIZE );
    imshow( "Gx", Ix);
    namedWindow("Gy", WINDOW_AUTOSIZE );
    imshow( "Gy", Iy);
    namedWindow("Gxy", WINDOW_AUTOSIZE );
    imshow( "Gxy", Ixy);
    namedWindow("Sxy", WINDOW_AUTOSIZE );
    imshow( "Sxy", Score);

    waitKey(0);

    return 0;
}
