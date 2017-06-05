#include <vector>
#include <iostream>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/viz.hpp"

using namespace std;
using namespace cv;

const float inlier_threshold = 152.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio


#define IMAGE1 "image.png"
#define IMAGE2 "image2.png"

//#define IMAGE1 "batinria0.tif"
//#define IMAGE2 "batinria1.tif"


void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)  { 

//this function automatically gets rid of points for which tracking fails

  vector<float> err;                    
  Size winSize=Size(15,15);                                                                                             
  TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
    

  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  Point2f pt = points2.at(i- indexCorrection);
        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))    {
              if((pt.x<0)||(pt.y<0))    {
                status.at(i) = 0;
              }
              points1.erase (points1.begin() + (i - indexCorrection));
              points2.erase (points2.begin() + (i - indexCorrection));
              indexCorrection++;
        }
     }
}

void featureDetection(Mat img_1, vector<Point2f>& points1, vector<KeyPoint> &keypoints_1)  {   //uses FAST as of now, modify parameters as necessary
  //vector<KeyPoint> keypoints_1;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  KeyPoint::convert(keypoints_1, points1, vector<int>());

}



int main(void) {
    Mat img1 = imread(IMAGE1, IMREAD_GRAYSCALE);
    Mat img2 = imread(IMAGE2, IMREAD_GRAYSCALE);

    vector<KeyPoint> keypoints_1, keypoints_2;

    vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
    featureDetection(img1, points1, keypoints_1);        //detect features in img_1
    featureDetection(img2, points2, keypoints_2);        //detect features in img_1

    vector<uchar> status;
    featureTracking(img1,img2,points1,points2, status); //track those features to img_2

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
    cout<< "M:" << M << endl;
    cout<< "Mt:" << Mt << endl;
    cout<< "Mt*M:" << MM << endl;

    Mat eigenVal, eigenVec;
    eigen(MM, eigenVal, eigenVec);
    cout << "Soluton:\n"; 
    //cout << "EigenVal: " << eigenVal << endl;
    //cout << "EigenVec: " << eigenVec << endl;
    Mat lambda = eigenVec.col(0);
    Mat gamma  = eigenVec.col(points1.size() );
    //return 0;
    //cout << points1.size();

    cout << "Lambda: " << lambda << endl;
    //cout << "Gamma: " << gamma << endl;
    printType(lambda, "lambda");
    vector<Point3f> points_proj;
    for (int i=0; i< points1.size();i++) { 
        points_proj.push_back(Point3f(points1[i].x, points2[i].y, ( (float) lambda.at<double>(0,i) )*100 ));   
    }
    

    viz::WCloud cloud_widget = viz::WCloud( points_proj, cv::viz::Color::green() );
    cloud_widget.setRenderingProperty( cv::viz::POINT_SIZE, 2 );
    viz::Viz3d trajectoryWindow("show");

    trajectoryWindow.showWidget("point_cloud", cloud_widget);
   
    trajectoryWindow.spin();
    waitKey(0);
    return 0;
}

