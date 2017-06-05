#include <iostream>
#include <unistd.h>
#include "opencv2/core.hpp"
#include "opencv2/viz.hpp"

using namespace cv;
using namespace std;


cv::Point3f mult(cv::Mat a, cv::Mat b)
{
    cv::Mat_<float> dst = b*a;

    //printf("%f %f %f, %f %f %f\n", p.x, p.y, p.z, dst(0,0), dst(0,1), dst(0,2));
    float x = dst(0,0)/dst(2,0);
    float y = dst(1,0)/dst(2,0);
    return cv::Point3f(x, y, 1);//dst(2,0)); 
}
Mat point4(float a, float b, float c, float d) {
    cv::Mat_<float> src(4/* rows*/,1 /*  cols */); 

    src(0,0)=a; 
    src(1,0)=b; 
    src(2,0)=c; 
    src(3,0)=d;

    return src; 
} 
void applyMat(Mat m, vector<Mat> &vec) {
    for (int i=0; i< vec.size(); i++) {
        vec[i] = m*vec[i];
    }
}

int main()
{
    cout << "Built with OpenCV " << CV_VERSION << endl;

    FILE* points_file;
    points_file = fopen("model.off", "r");
    int points_count, triangles_count, zero;
    float m_x = 0, m_z = 0, m_y = 0; 
    fscanf(points_file, "%d %d %d", &points_count, &triangles_count, &zero);
    vector<Mat> point_cloud;
    vector<int> triangle;
    printf("reading points: %d\n", points_count);
    viz::Viz3d trajectoryWindow("show");
    float x,y,z;
    for (int i=0; i< points_count;i++) {
        fscanf(points_file, "%f %f %f", &x, &y, &z);
        Mat homogeneus_point = point4(x, y, z, 1);
        point_cloud.push_back( homogeneus_point );
    }

    int d, a, b, c;
    for (int i=0; i< triangles_count;i++) {
        fscanf(points_file, "%d %d %d %d", &d, &a, &b, &c);
        triangle.push_back(d);
        triangle.push_back(a);
        triangle.push_back(b);
        triangle.push_back(c);
    }

    float moveToCameraCoordinates[] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1};
 
    Mat trans = Mat(4,4, CV_32F, moveToCameraCoordinates);
    applyMat(trans, point_cloud);
    float f = 1.0;
    float k[] = { 
        f, 0, 0, 0, 
        0, f, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 0};
    Mat K = Mat(4,4, CV_32F, k);
    vector<Point3f> points_proj;
    for (int i=0; i< point_cloud.size();i++) { 
        points_proj.push_back(mult(point_cloud[i], K));   
    }
    

    viz::WCloud cloud_widget = viz::WCloud( points_proj, cv::viz::Color::green() );
    cloud_widget.setRenderingProperty( cv::viz::POINT_SIZE, 2 );

    trajectoryWindow.showWidget("point_cloud", cloud_widget);
   
    trajectoryWindow.spin();
    return 0;
}
