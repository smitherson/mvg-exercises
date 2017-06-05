#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include "opencv2/viz.hpp"

using namespace std;
using namespace cv;

const float inlier_threshold = 125.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
//#define IMAGE1 "image.png"
//#define IMAGE2 "image2.png"

#define IMAGE1 "batinria0.tif"
#define IMAGE2 "batinria1.tif"

void drawPoints(Mat img, vector<Point2f> points, Mat& res) {
    res = img.clone();
    for (int i=0;i< points.size();i++) {
        circle( res,
         points[i],
         2.0,
         Scalar( 0, 0, 255 ));
    }
}


int main(void)
{
    Mat img1 = imread(IMAGE1, IMREAD_GRAYSCALE);
    Mat img2 = imread(IMAGE2, IMREAD_GRAYSCALE);


    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;

    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);

    vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
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
            if (points1.size() == 10) break;
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
    ///////////////////////
    
    Mat res1, res2;
    drawPoints(img1, points1, res1);
    drawPoints(img2, points2, res2);
    imshow("matches1", res1);
    imshow("matches2", res2);
    //waitKey(0);
    //return 0;

    Mat E, R, t, mask;
    E = findEssentialMat(points2, points1);// focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points2, points1, R, t);// focal, pp, mask);
    //cout << "E: " << E << endl;
    cout << "R: " << R << endl;
    cout << "t: " << t << endl;
    //cout << "x0: " << pointToMat(points2[0]) << endl;
    //printf("Matrix: %s %s \n",  type2str(hat( pointToMat(points2[0])).type()).c_str(), type2str(t.type()).c_str());

    //cout << "2 : " << hat( pointToMat(points2[0]))*t << endl;
    //cout << "1 : " << hat( pointToMat(points2[0]))*R*pointToMat(points1[0]) << endl;

    printType(R,"R type:");
    printType(t,"t type:");

    Mat M = Mat::zeros(3*points1.size(), points1.size()+1, CV_64F);
    assert(points1.size() == points2.size());
    for (int i=0;i<points1.size();i++) {
        Mat val1 = hat( pointToMat(points2[i]))*R*pointToMat(points1[i]);
        Mat val2 = hat( pointToMat(points2[i]))*t;

        //cout << "val 1:" << val1 << endl; 
        //cout << "val 2:" << val2 << endl;        
        //cout << endl;
        M.at<double>(3*i, i) = val1.at<double>(0,0);
        M.at<double>(3*i+1, i) = val1.at<double>(1,0);
        M.at<double>(3*i+2, i) = val1.at<double>(2,0);

        M.at<double>(3*i, points1.size()) = val2.at<double>(0,0);
        M.at<double>(3*i+1, points1.size()) = val2.at<double>(1,0);
        M.at<double>(3*i+2, points1.size()) = val2.at<double>(2,0);
    }
    Mat Mt;
    transpose(M,Mt);
    Mat MM =  Mt*M;
    cout<< "M:" << M << endl << endl;
    cout<< "Mt:" << Mt << endl << endl;
    cout<< "Mt*M:" << MM << endl << endl;

    Mat eigenVal, eigenVec;
    eigen(MM, eigenVal, eigenVec);
    cout << "Soluton:\n"; 
    cout << "EigenVal: " << eigenVal << endl << endl;
    cout << "EigenVec: "<< eigenVec << endl << endl;
    Mat lambda = eigenVec.col(0);
    Mat gamma  = eigenVec.col(points1.size() );
    //return 0;
    //cout << points1.size();

    cout << "Lambda: " << lambda << endl << endl;
    cout << "Gamma: " << gamma << endl << endl;
    printType(lambda, "lambda");
    vector<Point3f> points_proj;
    for (int i=0; i< points1.size();i++) { 
        points_proj.push_back(Point3f(points1[i].x, points2[i].y, ( (float) lambda.at<double>(0,i) ) ));   
    }
    

    viz::WCloud cloud_widget = viz::WCloud( points_proj, cv::viz::Color::green() );
    cloud_widget.setRenderingProperty( cv::viz::POINT_SIZE, 2 );
    viz::Viz3d trajectoryWindow("show");

    trajectoryWindow.showWidget("point_cloud", cloud_widget);
   
    trajectoryWindow.spin();
    waitKey(0);
    return 0;


    /////////////
    return 0;
}
