#include <vector>
#include <iostream>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/viz.hpp"
#include "../common.h"

using namespace std;
using namespace cv;

//#define IMAGE1 "image.png"
//#define IMAGE2 "image2.png"

#define IMAGE1 "batinria0.tif"
#define IMAGE2 "batinria1.tif"


//% Compute correct combination of R and T and reconstruction of 3D points
void reconstruction(Mat R, Mat T, vector<Point2f> points1, vector<Point2f> points2) {

    int nPoints = points1.size();
    Mat M = Mat::zeros(3*nPoints, nPoints+1, CV_32F );
    for (int i = 0; i<nPoints;i++) {
        Mat x2_hat = hat( pointToMat(points2[i]));
        Mat x1 = pointToMat(points1[i]);
        Mat val1 = x2_hat * R * x1;
        Mat val2 = x2_hat * T;

        M.at<float>(3*i, i) = val1.at<float>(0,0);
        M.at<float>(3*i+1, i) = val1.at<float>(1,0);
        M.at<float>(3*i+2, i) = val1.at<float>(2,0);

        M.at<float>(3*i, nPoints) = val2.at<float>(0,0);
        M.at<float>(3*i+1, nPoints) = val2.at<float>(1,0);
        M.at<float>(3*i+2, nPoints) = val2.at<float>(2,0);
    }

    //% Get depth values (eigenvector to the smallest eigenvalue of M'M):
      Mat Mt;
    transpose(M,Mt);
    Mat MM =  Mt*M;

    Mat eigenVal, eigenVec;

    eigen(MM, eigenVal, eigenVec);
    Mat lambda = eigenVec.row(eigenVec.rows-1);
    float gamma  = eigenVec.at<float>(nPoints,nPoints);

    cout << "Lambda: " << lambda << endl;
    cout << "Gamma: " << gamma << endl;
    vector<Point3f> points_proj;
    for (int i=0; i< points1.size();i++) { 
        points_proj.push_back(Point3f(points1[i].x, points1[i].y, lambda.at<float>(0,i)  ));   
    }
    

    viz::WCloud cloud_widget = viz::WCloud( points_proj, cv::viz::Color::green() );
    cloud_widget.setRenderingProperty( cv::viz::POINT_SIZE, 2 );
    viz::Viz3d trajectoryWindow("show");

    trajectoryWindow.showWidget("point_cloud", cloud_widget);
   
    trajectoryWindow.spin();
 
}



void getPredefinedPoints(vector<Point2f> &v1, vector<Point2f> &v2) {
    float x1[] = {10, 92, 8, 92, 289, 354, 289, 353,
        69, 294, 44, 336};

    float y1[] = {232, 230, 334, 333, 230, 278,
        340, 332, 90, 149, 475, 433};
 
    float x2[] = {123, 203, 123, 202, 397, 472, 398, 472,
        182, 401, 148, 447};

    float y2[] = {239, 237, 338, 338, 236, 286,
        348, 341, 99, 153, 471, 445};
    
    for (int i=0;i<12;i++) {
        v1.push_back(Point2f(x1[i],y1[i]));
        v2.push_back(Point2f(x2[i],y2[i]));
    }
}
const float inlier_threshold = 125.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

int main() {
    Mat img1 = imread(IMAGE1, IMREAD_GRAYSCALE);
    Mat img2 = imread(IMAGE2, IMREAD_GRAYSCALE);
    vector<Point2f> points1, points2; 
    getPredefinedPoints(points1, points2);
    /*vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;

    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);

    vector<KeyPoint> matched1, matched2, inliers1, inliers2;
    vector<DMatch> good_matches;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }

    for(unsigned i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;

        col /= col.at<double>(2);
        double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                            pow(col.at<double>(1) - matched2[i].pt.y, 2));

        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
            points1.push_back(matched1[i].pt);
            points2.push_back(matched2[i].pt);
            if (points1.size() == 1000) break;
        }
    }
    Mat res;
    drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
    imshow("res.png", res);
    //waitKey(0);
    double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
    cout << "A-KAZE Matching Results" << endl;
    cout << "*******************************" << endl;
    cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
    cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
    cout << "# Matches:                            \t" << matched1.size() << endl;
    cout << "# Inliers:                            \t" << inliers1.size() << endl;
    cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
    cout << endl;
*/
//////////////
    float k1[] = {844.310547, 0, 243.413315, 0, 1202.508301, 281.529236, 0, 0, 1};
    float k2[] = {852.721008, 0, 252.021805, 0, 1215.657349, 288.587189, 0, 0, 1};
    Mat K1 = Mat(3,3, CV_32F, k1);
    Mat K2 = Mat(3,3, CV_32F, k2);
    K1 = K1.inv();
    K2 = K2.inv();
    //printVector(points1);
    for (int i=0;i < points1.size();i++ ){
        Mat rp;
        //cout << "before P: " << points1[i] << endl;
        rp = K1*pointToMat(points1[i]);
        //cout << pointToMat(points1[i]) << endl;
        points1[i].x = rp.at<float>(0,0);
        points1[i].y = rp.at<float>(0,1);
        //cout << "after P: " << points1[i] << endl;

        rp = K2*pointToMat(points2[i]);
        points2[i].x = rp.at<float>(0,0);
        points2[i].y = rp.at<float>(0,1);
    }
    //printVector(points1);
    Mat chi = Mat::zeros(points1.size(),9, CV_32F);
    vector<Mat> krons;
    for (int i = 0;i<points1.size();i++) {
        Mat temp =  kron( pointToMat( points1[i]), pointToMat( points2[i]));
        //cout << "Kron: " << temp << endl << endl;
        krons.push_back( temp);
    }
    hconcat(krons, chi);
    transpose(chi,chi); 
    //cout << "Chi: "; 
    //printMat(chi);
    //rank_chi = rank(chi)
    SVD svd = SVD(chi);
    Mat v; 
    transpose( svd.vt, v); 
    vector<float> Ev;
    v.col(8).copyTo(Ev);
    
    //cout<< "Ev: "; printVector(Ev);
    //cout << "Es: " <<   << endl;
    Mat E = Mat(3,3, CV_32F, &Ev[0]);
    transpose(E,E);
    //cout << "E: "; printMat(E);

    SVD Esvd = SVD(E);
    Mat EsvdV; 
    transpose( Esvd.vt, EsvdV);

    //cout << "Esvd.u: ";  printMat(Esvd.u);
    //cout << "Esvd.v: ";  printMat(EsvdV);
   
    if (determinant(Esvd.u) < 0 || determinant(EsvdV) < 0) {
        Esvd = SVD(-E);
        //cout << "changed sing\n";
    }

    Mat D = Mat::zeros(3,3, CV_32F);
    D.at<float>(0,0) = 1;
    D.at<float>(1,1) = 1;
    D.at<float>(2,2) = 0;

    E = Esvd.u*D*Esvd.vt;
    Mat U = Esvd.u;
    Mat Vr = transpose(Esvd.u);

    Mat Vt = Esvd.vt;
    //cout << "project E: "; 
    //printMat(E);
    //cout << "OpenCV E:"; 
    //printMat( findEssentialMat(points1, points2));


    float rz1[] = {0, -1, 0, 1, 0, 0, 0, 0, 1};
    float rz2[] = {0, 1, 0, -1, 0, 0, 0, 0, 1};
    Mat Rz1 = Mat(3,3,CV_32F, rz1);
    Mat Rz2 = Mat(3,3,CV_32F, rz2);    

    Mat R1 = U * transpose( Rz1) * Vt;
    Mat R2 = U * transpose( Rz2) * Vt;
    Mat Ut = transpose(U);
    Mat T_hat1 = U * Rz1 * D * Ut;
    Mat T_hat2 = U * Rz2 * D * Ut;

    cout << "R1: "; printMat(R1);
    //cout << "R2: "; printMat(R2);

    //% Translation belonging to T_hat
    Mat T1 = Mat::zeros(3,1, CV_32F);
    Mat T2 = Mat::zeros(3,1, CV_32F);

    T1.at<float>(0,0) = -T_hat1.at<float>(1,2);
    T1.at<float>(1,0) = T_hat1.at<float>(0,2); 
    T1.at<float>(2,0) = -T_hat1.at<float>(0,1);

    T2.at<float>(0,0) = -T_hat2.at<float>(1,2);
    T2.at<float>(1,0) = T_hat2.at<float>(0,2); 
    T2.at<float>(2,0) = -T_hat2.at<float>(0,1);

    cout << "T1: "; 
    printMat(T1);
    //cout << "T2: "; 
    //printMat(T2);

    reconstruction(R1, T1, points1, points2);
    reconstruction(R1, T2, points1, points2);
    reconstruction(R2, T1, points1, points2);
    reconstruction(R2, T2, points1, points2);

    return 0;
}

/*  


% Compute scene reconstruction and correct combination of R and T:
reconstruction(R1,T1,x1,y1,x2,y2,nPoints);
reconstruction(R1,T2,x1,y1,x2,y2,nPoints);
reconstruction(R2,T1,x1,y1,x2,y2,nPoints);
reconstruction(R2,T2,x1,y1,x2,y2,nPoints);

end








% ================
% Hat-function
function A = hat(v)
    A = [0 -v(3) v(2) ; v(3) 0 -v(1) ; -v(2) v(1) 0];
end



% ================
% function getpoints
function [x1,y1,x2,y2] = getpoints(image1,image2,nPoints)

x1 = zeros(nPoints,1);
y1 = zeros(nPoints,1);
x2 = zeros(nPoints,1);
y2 = zeros(nPoints,1);

% Click points in image1:
% Can be done without for-loop: ginput(nPoints)
figure; imshow(uint8(image1));
hold on;
for i = 1:nPoints
    [x,y] = ginput(1);
    x1(i) = double(x);
    y1(i) = double(y);
    plot(x, y, 'r+');
end
hold off;


% Click points in image2:
figure; imshow(uint8(image2));
hold on;
for i = 1:nPoints
    [x,y] = ginput(1);
    x2(i) = double(x);
    y2(i) = double(y);
    plot(x, y, 'r+');
end
hold off;

end



% ================
% function getpoints2  --> points already defined
function [x1,y1,x2,y2] = getpoints2()

x1 = [
   10
   92
    8
   92
  289
  354
  289
  353
   69
  294
   44
  336
  ];

y1 = [ 
  232
  230
  334
  333
  230
  278
  340
  332
   90
  149
  475
  433
    ];
 
x2 = [
  123
  203
  123
  202
  397
  472
  398
  472
  182
  401
  148
  447
    ];

y2 = [ 
  239
  237
  338
  338
  236
  286
  348
  341
   99
  153
  471
  445
    ];

end
*/
