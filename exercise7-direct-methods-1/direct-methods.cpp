#include <vector>
#include <iostream>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <limits>       // std::numeric_limits
#include <unistd.h>
#include "../common.h"

using namespace std;
using namespace cv;


/*#define IMAGE1 "/home/smith/Work/sdvo-phone/test-footy/rgb/1341847980.722988.png"
#define IMAGE2 "/home/smith/Work/sdvo-phone/test-footy/rgb/1341847982.998783.png"
#define IMAGE1_DEPTH "/home/smith/Work/sdvo-phone/test-footy/depth/1341847980.723020.png"
#define IMAGE2_DEPTH "/home/smith/Work/sdvo-phone/test-footy/depth/1341847982.998830.png"
#define k_values { 535.4, 0, 320.1, 0, 539.2, 247.6, 0, 0, 1}
*/

#define IMAGE1 "../rgb/1305031102.175304.png"
#define IMAGE2 "../rgb/1305031102.275326.png"
#define IMAGE1_DEPTH "../depth/1305031102.160407.png"
#define IMAGE2_DEPTH "../depth/1305031102.262886.png"
#define k_values { 517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1}



#define LVL 5

float dNaN = numeric_limits<float>::quiet_NaN();
void deriveErrAnalitic(Mat_<float> IRef, Mat_<float> DRef, Mat_<float> I, Mat_<float> xi, Mat_<float> K, Mat_<float>& Jac, Mat& residual, int lvl ) {
    Mat T = se3Exp(xi);
    Mat R = T(cv::Rect(0,0,3,3));
    Mat t = T(cv::Rect(3,0,1,3));

    Mat RKInv = R * K.inv();

    Mat_<float> Img =  Mat(IRef.rows, IRef.cols,CV_32F, dNaN);
    Mat_<float> dyI, dxI;
    calculateGradients<float>(I, dxI, dyI);
    Mat_<float> dxImg = Mat(IRef.rows, IRef.cols, CV_32F, dNaN);
    Mat_<float> dyImg = Mat(IRef.rows, IRef.cols, CV_32F, dNaN);

    Mat_<float> xImg =  Mat(IRef.rows, IRef.cols,CV_32F, dNaN);
    Mat_<float> yImg =  Mat(IRef.rows, IRef.cols,CV_32F, dNaN);

    Mat_<float> xp = Mat(IRef.rows, IRef.cols, CV_32F, dNaN);
    Mat_<float> yp = Mat(IRef.rows, IRef.cols, CV_32F, dNaN);
    Mat_<float> zp = Mat(IRef.rows, IRef.cols, CV_32F, dNaN);
 
    for (int x=0; x< IRef.rows; x++) {
        for (int y=0; y< IRef.cols; y++) {
            if (DRef.at<float>(x,y) > 0) {

                Mat_<float> p = Mat( Point3d(x,y,1)) * DRef(x,y);
                Mat_<float> pTrans =  ( RKInv * p + t );
                if (pTrans(2, 0) > 0 ) {
                    Mat_<float> pTransProj =  K*pTrans;

                        xImg(x,y) = pTransProj(0,0)/pTransProj(2,0);
                        yImg(x,y) = pTransProj(1,0)/pTransProj(2,0);

                        xp(x, y) = pTrans(0,0);
                        yp(x, y) = pTrans(1,0);
                        zp(x, y) = pTrans(2,0);
                } 
            }
        }
    }

    remap(I, Img, yImg, xImg, CV_INTER_LINEAR, BORDER_CONSTANT, dNaN);
        
    remap(dxI, dxImg, yImg, xImg, CV_INTER_LINEAR, BORDER_CONSTANT, dNaN);
    remap(dyI, dyImg, yImg, xImg, CV_INTER_LINEAR, BORDER_CONSTANT, dNaN);

    dxImg = K(0,0) * dxImg.reshape(1, dxImg.rows*dxImg.cols);
    dyImg = K(1,1) * dyImg.reshape(1, dyImg.rows*dyImg.cols); 
   
    xp = xp.reshape(1, xp.rows*xp.cols);
    yp = yp.reshape(1, yp.rows*yp.cols);
    zp = zp.reshape(1, zp.rows*zp.cols);

    Jac = Mat::zeros(Img.cols * Img.rows, 6, CV_32F);
    
    for (int i=0;i<zp.rows;i++) {
            Jac(i,0) = dxImg(i,0) / zp(i,0);
            Jac(i,1) = dyImg(i,0) / zp(i,0);
            
            Jac(i,2) = - (dxImg(i,0) * xp(i,0) + dyImg(i,0) * yp(i,0)) / (zp(i,0) * zp(i,0));

            Jac(i,3) = - (dxImg(i,0) * xp(i,0) * yp(i,0)) / (zp(i,0) * zp(i,0)) - dyImg(i,0) * (1 + (yp(i,0) / zp(i,0))*(yp(i,0) / zp(i,0)));
            Jac(i,4) = + dxImg(i,0) * (1 + (xp(i,0) / zp(i,0))*(xp(i,0) / zp(i,0))) + (dyImg(i,0) * xp(i,0) * yp(i,0)) / (zp(i,0) * zp(i,0));
            Jac(i,5) = (- dxImg(i,0) * yp(i,0) + dyImg(i,0) * xp(i,0)) / zp(i,0);
    }
   
    Jac = -Jac;
    Mat err = IRef - Img;
    Mat_<uchar> errU8;
    err.convertTo(errU8, CV_8U);
    upscaleImage<uchar>(errU8, lvl);
    for (int row = 0; row<errU8.rows;row++) {
        for(int col = 0; col<errU8.cols;col++) {
            if ( errU8(row,col) > 250 ) {
                errU8(row,col) = 0;
            }
        }
    }
    imshow("err", errU8);
    waitKey(0);
    residual =  err.reshape(1, err.rows*err.cols);
}

int main() {
    int lvl = LVL;

    float k[] = k_values;
    Mat_<float> K = Mat(3,3, CV_32F, k);
    Mat img1 = imread(IMAGE1, IMREAD_GRAYSCALE);
    Mat img2 = imread(IMAGE2, IMREAD_GRAYSCALE);

    imshow("start ", img1-img2);  
    imshow("img1: ", img1);  
    imshow("img2: ", img2);  

    Mat img1_depth = imread(IMAGE1_DEPTH, IMREAD_ANYDEPTH);
    Mat img2_depth = imread(IMAGE2_DEPTH, IMREAD_ANYDEPTH);

    img1.convertTo(img1, CV_32F);
    img1_depth.convertTo(img1_depth, CV_32F);
    img2.convertTo(img2, CV_32F);
    img2_depth.convertTo(img2_depth, CV_32F);
    Mat xi = Mat::zeros(1,6,CV_32F);
    //float xi_solution[] = xi_values; 
    //Mat xi = Mat(1,6,CV_32F, xi_solution); 

    Mat IRef = img1.clone();
    Mat ITwo = img2.clone();
    Mat DRef = img1_depth.clone();
    Mat_<float> Klvl = K.clone();

    for (;lvl>1;lvl--) {
        Mat IRef = img1.clone();
        Mat ITwo = img2.clone();
        Mat DRef = img1_depth.clone();
        Mat_<float> Klvl = K.clone();

        downscaleImage<float>(IRef, lvl);
        downscaleDepth<float>(DRef, lvl);
        downscaleImage<float>(ITwo, lvl);
        downscaleK(Klvl, lvl);
        cout << lvl << endl;
       
        Mat_<float> Jac;
        Mat_<float> residual;

        Scalar errScalar;
        float err;
        float errLast = 99999;
        for (int i=0; i<10; i++) {

            deriveErrAnalitic(IRef, DRef, ITwo, xi, Klvl, Jac, residual, lvl);
           
            for (int row = 0; row < Jac.rows; row++) {
                for(int col = 0; col < Jac.cols; col++) {
                    if ( isnan(Jac(row,col)) || isnan(residual(row,0) || abs( Jac(row,col)) > 10000  )) {
                        for(int col2 = 0; col2 < Jac.cols; col2++) {
                            Jac(row, col2) =  (float)0;
                        }
                        residual(row,0) = 0;
                    }
                }
            }
            
            Mat upd = - ( transpose(Jac) * Jac).inv(DECOMP_SVD)* (transpose(Jac)* residual);
            xi = se3Log(se3Exp(upd) * se3Exp(xi));
            cout <<"updated Xi: "; printMat(xi);
            float sumErr = 0;
            Mat_<float> resid2 = residual.clone();
            for (int item=0; item<resid2.cols; item++) {
                resid2(0, item) *= resid2(0, item); 
                sumErr += resid2(0,item);
            }
            errScalar = mean(resid2);
            err = errScalar.val[0];
            cout << "Error: " << err << endl; 
            cout << "Error sum: " << sumErr << endl;
            if(err / errLast > 0.99) {
                break;
            }
            errLast = err;
        }
    }
    waitKey(0);

    return 0;
}

