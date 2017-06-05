#include <vector>
#include <iostream>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
//#include "opencv2/viz.hpp"
#include <limits>       // std::numeric_limits
#include <unistd.h>
#include "../common.h"

using namespace std;
using namespace cv;
/*
#define IMAGE1 "./rgb/1341847980.722988.png"
#define IMAGE2 "./rgb/1341847982.998783.png"
#define IMAGE1_DEPTH "./depth/1341847980.723020.png"
#define IMAGE2_DEPTH "./depth/1341847982.998830.png"
#define k_values { 535.4, 0, 320.1, 0, 539.2, 247.6, 0, 0, 1}
*/
#define IMAGE1 "../rgb/1305031102.175304.png"
#define IMAGE2 "../rgb/1305031102.275326.png"
#define IMAGE1_DEPTH "../depth/1305031102.160407.png"
#define IMAGE2_DEPTH "../depth/1305031102.262886.png"
#define k_values { 535.4, 0, 320.1, 0, 539.2, 247.6, 0, 0, 1}

#define LVL 5
Mat calcErr(Mat_<float> IRef_, Mat_<float> DRef, Mat_<float> I, Mat_<float> xi, Mat_<float> K, int lvl ) {
    Mat_<float> IRef = IRef_.clone();
    Mat T = se3Exp(xi);
    Mat R = T(cv::Rect(0,0,3,3));
    Mat t = T(cv::Rect(3,0,1,3));
    Mat RKInv = R * K.inv();

    //float dNaN = numeric_limits<float>::quiet_NaN();
    Mat_<float> Img = Mat(IRef.rows, IRef.cols,CV_32F, cv::Scalar(0));

    for (int x=0; x< IRef.rows; x++) {
        for (int y=0; y< IRef.cols; y++) {
            //% point in reference image. note that the pixel-coordinates of the
            //% point (1,1) are actually (0,0).
            if (DRef.at<float>(x,y) > 0) {

                Mat_<float> p = Mat( Point3d(x,y,1)) * DRef(x,y);
                //cout << "p: "; printMat(p);
                //% transform to image (unproject, rotate & translate)
                Mat_<float> pTrans = K * ( RKInv * p + t );
                //cout << "pTrans: "; printMat(pTrans);
                //% if point is valid (depth > 0), project and save result.
                if (pTrans(2, 0) > 0 ) {
                    int newX = (int)round(pTrans(0,0)/pTrans(2,0));
                    int newY = (int)round(pTrans(1,0)/pTrans(2,0));
                    
                    if (newX >= 0 && newX < Img.rows && newY >= 0 && newY < Img.cols) {
                        Img(newX,newY) = I(x,y);
                    }
                } else {
                     Img(x,y) = 0; 
                     //IRef(x,y) = 0; 
                }

            } else { 
                Img(x,y) = 0;
                //IRef(x,y) = 0; 
            }
        }
    }


    Mat err = IRef - Img;
    Mat err8U;
    err.convertTo(err8U, CV_8U);
    upscaleImage<uchar>(err8U, lvl);
    imshow("err", err8U);
    usleep(100);
    waitKey(1);
    Mat errLine =  err.reshape(1, err.rows*err.cols);
    errLine.convertTo(errLine, CV_32F);

    return errLine.clone();
}

void deriveErrNumeric(Mat IRef, Mat DRef, Mat I, Mat_<float> xi, Mat_<float> K, Mat& Jac, Mat& residual, int lvl ) {
     
    //translation
    float eps = 0.00195;
    Jac = Mat::zeros(I.rows * I.cols, 6, CV_32F);
    residual = calcErr(IRef,DRef,I,xi,K,lvl);
    for (int j=0;j<6;j++) {
        Mat_<float> epsVec = Mat::zeros(1,6, CV_32F);
        epsVec(0,j) = eps;
        //cout << "epsVec: "; printMat(epsVec);
       
        //% MULTIPLY epsilon from left onto the current estimate.
        Mat tmp = se3Exp(epsVec) * se3Exp(xi);
        Mat xiPerm =  se3Log(tmp);
        Jac.col(j) = (calcErr(IRef,DRef,I,xiPerm,K,lvl) - residual) / eps;

        //printMat( xiPerm);
    }
    //printMat(Jac);
}

void deriveErrAnalitic(Mat_<float> IRef, Mat_<float> DRef, Mat_<float> I, Mat_<float> xi, Mat_<float> K, Mat_<float>& Jac, Mat& residual, int lvl ) {
    Mat T = se3Exp(xi);
    Mat R = T(cv::Rect(0,0,3,3));
    Mat t = T(cv::Rect(3,0,1,3));

    Mat RKInv = R * K.inv();

    //float dNaN = numeric_limits<float>::quiet_NaN();
    Mat_<float> Img = Mat(IRef.rows, IRef.cols,CV_32F, cv::Scalar(0));
    Mat_<float> dyI, dxI;
    calculateGradients<float>(I, dxI, dyI);
  
    Mat_<float> dxImg = Mat(IRef.rows, IRef.cols,CV_32F, cv::Scalar(0));
    Mat_<float> dyImg = Mat(IRef.rows, IRef.cols,CV_32F, cv::Scalar(0));

    Mat_<float> xp = Mat(IRef.rows, IRef.cols,CV_32F, cv::Scalar(0));
    Mat_<float> yp = Mat(IRef.rows, IRef.cols,CV_32F, cv::Scalar(0));
    Mat_<float> zp = Mat(IRef.rows, IRef.cols,CV_32F, cv::Scalar(0));
 
    for (int x=0; x< IRef.rows; x++) {
        for (int y=0; y< IRef.cols; y++) {
            //% point in reference image. note that the pixel-coordinates of the
            //% point (1,1) are actually (0,0).
            if (DRef.at<float>(x,y) > 0) {

                Mat_<float> p = Mat( Point3d(x,y,1)) * DRef(x,y);
                //cout << "p: "; printMat(p);
                //% transform to image (unproject, rotate & translate)
                Mat_<float> pTrans = K * ( RKInv * p + t );
                //cout << "pTrans: "; printMat(pTrans);
                //% if point is valid (depth > 0), project and save result.
                if (pTrans(2, 0) > 0 ) {
                    int newX = (int)round(pTrans(0,0)/pTrans(2,0));
                    int newY = (int)round(pTrans(1,0)/pTrans(2,0));
                    if (newX >= 0 && newX < Img.rows && newY >= 0 && newY < Img.cols) {
                        Img(newX,newY) = I(x,y);
                        dxImg(newX,newY) = dxI(x,y);
                        dyImg(newX,newY) = dyI(x,y);
                        xp(newX, newY) = pTrans(0,0);
                        yp(newX, newY) = pTrans(1,0);
                        zp(newX, newY) = pTrans(2,0);
                    }
                } else {
                     //Img(x,y) = 0; 
                }

            } else { 
                //Img(x,y) = 0;
            }
        }
    }

    Mat tmp; /*
    dxImg.convertTo(tmp, CV_8U);
    upscaleImage<uchar>(tmp, lvl);
    imshow("dxImg", tmp);   

    (dyImg.convertTo(tmp, CV_8U);
    upscaleImage<uchar>(tmp, lvl);
    imshow("dyImg", tmp);   
    
    Img.convertTo(tmp, CV_8U);
    upscaleImage<uchar>(tmp, lvl);
    imshow("Img", tmp);   
    
    waitKey(0);*/ 
    dxImg = K(0,0) * dxImg.reshape(1, dxImg.rows*dxImg.cols);
    dyImg = K(1,1) * dyImg.reshape(1, dyImg.rows*dyImg.cols); 

    //% 2.: get warped 3d points (x', y', z').
    xp = xp.reshape(1, xp.rows*xp.cols);
    yp = yp.reshape(1, yp.rows*yp.cols);
    zp = zp.reshape(1, zp.rows*zp.cols);


    //% 3. direct implementation of kerl2012msc.pdf Eq. (4.14):

    Jac = Mat::zeros(Img.cols * Img.rows, 6, CV_32F);

    for (int i=0;i<zp.rows;i++) {
        if (zp(i,0) > 1 && dxImg(i,0) > 5 && dyImg(i,0) > 5) {
            Jac(i,0) = dxImg(i,0) / zp(i,0);
            Jac(i,1) = dyImg(i,0) / zp(i,0);
            Jac(i,2) = - (dxImg(i,0) * xp(i,0) + dyImg(i,0) * yp(i,0)) / (zp(i,0) * zp(i,0));

            Jac(i,3) = - (dxImg(i,0) * xp(i,0) * yp(i,0)) / (zp(i,0) * zp(i,0)) - dyImg(i,0) * (1 + (yp(i,0) / zp(i,0))*(yp(i,0) / zp(i,0)));
            Jac(i,4) = + dxImg(i,0) * (1 + (xp(i,0) / zp(i,0))*(xp(i,0) / zp(i,0))) + (dyImg(i,0) * xp(i,0) * yp(i,0)) / (zp(i,0) * zp(i,0));
            Jac(i,5) = (- dxImg(i,0) * yp(i,0) + dyImg(i,0) * xp(i,0)) / zp(i,0);
        }
    }
    //i% invert jacobian: in kerl2012msc.pdf, the difference is defined the other
    //% way round, see (4.6).

    //printMat(Jac);
    //Jac = -Jac;
    Mat err = IRef - Img;
    Mat err8U;

    Img.convertTo(err8U, CV_8U);
    upscaleImage<uchar>(err8U, lvl);
    imshow("err",err8U);

    waitKey(1);
    residual =  err.reshape(1, err.rows*err.cols);
}

int main() {
    int lvl = LVL;

    float k[] = k_values;
    Mat_<float> K = Mat(3,3, CV_32F, k);
    Mat img1 = imread(IMAGE1, IMREAD_GRAYSCALE);
    Mat img2 = imread(IMAGE2, IMREAD_GRAYSCALE);

    imshow("orig 1: ", img1-img2);  
    //imshow("orig 2: ", img2);  

    Mat img1_depth = imread(IMAGE1_DEPTH, IMREAD_ANYDEPTH);
    Mat img2_depth = imread(IMAGE2_DEPTH, IMREAD_ANYDEPTH);

    img1.convertTo(img1, CV_32F);
    img1_depth.convertTo(img1_depth, CV_32F);
    img2.convertTo(img2, CV_32F);
    img2_depth.convertTo(img2_depth, CV_32F);
    Mat xi = Mat::zeros(1,6,CV_32F);
    //float xi_solution[] = xi_values; 
    //Mat xi = Mat(1,6,CV_32F, xi_solution); 

    /*Mat IRef = img1.clone();
    Mat ITwo = img2.clone();
    Mat DRef = img1_depth.clone();
    Mat_<float> Klvl = K.clone();*/


    //% use huber weights
    bool useHuber = true;

    //% exactly one of those should be true.
    bool useGN = true; //% Gauss-Newton
    bool useLM = false; //% Levenberg Marquad
    bool useGD = false; //% Gradiend descend

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
        /*
           Mat_<float> img1 = Mat(30, 40, CV_32F, iref);
           Mat_<float> img2 = Mat(30, 40, CV_32F, isecond); 
           Mat_<float> img1_depth = Mat(30, 40, CV_32F, dref); 
        */

        Mat_<float> Jac;
        Mat_<float> residual;

        Scalar errScalar;
        float lambda = 0.1;

        float err;
        float errLast = 1110;
        for (int i=0; i<10; i++) {

            //% calculate Jacobian of residual function (Matrix of dim (width*height) x 6)
            //deriveErrNumeric(IRef,DRef,ITwo,xi,Klvl, Jac, residual,lvl);
            deriveErrAnalitic(IRef, DRef, ITwo, xi, Klvl, Jac, residual, lvl);

            Mat_<float> huber = Mat::ones(residual.rows, 6, CV_32F);

            if (useHuber) {
                //% compute Huber Weights
                float huberDelta = 60;
                for (int j = 0; j < huber.rows; j++) {
                    //huber(abs(residual) > huberDelta) = huberDelta ./ abs(residual(abs(residual) > huberDelta));
                    if ( abs(residual(j,0)) > huberDelta ) {
                        huber.row(j) = huber.row(j)*(huberDelta / abs(residual(j,0))); 
                    }
                }
            }
    
            Mat upd;
            if(useGN) {
                upd = - ( transpose(Jac) * (huber.mul ( Jac))).inv(DECOMP_SVD)* (transpose(Jac)* ( residual.mul(huber.col(0))));
            }

            if(useGD) {
            //% do gradient descend
                upd = - transpose( Jac) * (residual.mul(huber.col(0)));
                cout << "norm(upd): " << norm(upd) << endl;
                upd = 0.001 * upd / norm(upd);  // % choose step size such that the step is always 0.001 long.
                
            }

            if (useLM) {
            //% do LM
                Mat H =  transpose( Jac) * ( Jac.mul( huber ));
                //cout << "H: "; printMat( Mat::diag( H.diag()));
                Mat diagonal = Mat::diag( H.diag());
                //cout << "H: "; printMat( );

                upd = - (H + lambda * diagonal).inv(DECOMP_SVD) * (transpose( Jac) * ( residual.mul(huber.col(0))));
            }

            //cout <<"upd: "; printMat(upd);

            //% MULTIPLY increment from left onto the current estimate.
            Mat lastXi;
            lastXi = xi.clone();

            xi = se3Log(se3Exp(upd) * se3Exp(xi));
            cout <<"updated Xi: "; printMat(xi);
            //% get mean and display
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
            //% break if no improvement

            if(useLM) {
                //err - errLast
                if(err >= errLast) {
                    lambda = lambda * 5;
                    xi = lastXi;

                    if(lambda > 5) {
                        break;
                    }
                    else {
                        lambda = lambda /1.5;
                    }
                }
            }
            if(useGN || useGD) {
                    if(err / errLast > 0.995) {
                        break;
                    }
             }
        }
    }
    /////--------------------------

    //upscaleImage(err,lvl);
    Mat T = se3Exp(xi);
    Mat R = T(cv::Rect(0,0,3,3));
    Mat t = T(cv::Rect(3,0,1,3));
    cout << "Final R: "; printMat( R);
    cout << "Final t: "; printMat(t);

    waitKey(0);
    /*cout<< "Image1: "; printType(img1);
    cout<< "Image1 depth: "; printType(img1_depth);
    
    vector<Point3f> points_proj;

    for (int i=0; i< img1.rows;i++) { 
        for(int j=0; j < img1.cols;j++) {
            char depth =  img1_depth.at<char>(i,j);
            if (depth > 0) {
                points_proj.push_back(Point3f(i, j, depth));
            }
        }
    }

    Mat colors(1,points_proj.size(), CV_8UC3);
    for (int i=0;i<points_proj.size();i++) {
        colors.at<Vec3b>(0,i) = img1.at<Vec3b>((int)points_proj[i].x, (int)points_proj[i].y);
    }

    viz::WCloud cloud_widget = viz::WCloud( points_proj, colors );
    cloud_widget.setRenderingProperty( cv::viz::POINT_SIZE, 2 );
    viz::Viz3d trajectoryWindow("show");

    trajectoryWindow.showWidget("point_cloud", cloud_widget);
   
    trajectoryWindow.spin();*/

    return 0;
}
