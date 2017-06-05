#include <iostream>
#include <unistd.h>
#include "opencv2/core.hpp"
#include "opencv2/viz.hpp"

using namespace cv;
using namespace std;


cv::Point3f mult(cv::Point3f& p, cv::Mat M)
{
    cv::Mat_<float> src(4/* rows*/,1 /*  cols */); 

    src(0,0)=p.x; 
    src(1,0)=p.y; 
    src(2,0)=p.z; 
    src(3,0) = 1;

    cv::Mat_<float> dst = M*src; //USE MATRIX ALGEBRA 
    //printf("%f %f %f, %f %f %f\n", p.x, p.y, p.z, dst(0,0), dst(0,1), dst(0,2));
    return cv::Point3f(dst(0,0),dst(1,0), dst(2,0)); 
}  
void applyMat(Mat m, vector<Point3f> &vec) {
    for (int i=0; i< vec.size(); i++) {
        vec[i] = mult(vec[i], m);
        // vec[i].x++;

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
    vector<Point3f> point_cloud;
    vector<int> triangle;
    printf("reading points: %d\n", points_count);
    viz::Viz3d trajectoryWindow("show");
    float x,y,z;
    for (int i=0; i< points_count;i++) {
        fscanf(points_file, "%f %f %f", &x, &y, &z);
        point_cloud.push_back(Point3f(x,y,z));
        m_x +=x; m_y+=y; m_z+=z;
    }

    int d, a, b, c;
    for (int i=0; i< triangles_count;i++) {
        fscanf(points_file, "%d %d %d %d", &d, &a, &b, &c);
        triangle.push_back(d);
        triangle.push_back(a);
        triangle.push_back(b);
        triangle.push_back(c);
    }
    m_x /= points_count;
    m_y /= points_count;
    m_z /= points_count;

    viz::WMesh mesh = viz::WMesh(point_cloud, triangle);
    trajectoryWindow.showWidget("point_cloud", mesh);

        float translation[] = {1,0,0,-m_x-0.5,
        0,1,0,-m_y-0.2,
        0,0,1,-m_z-0.1,
        0,0,0,1};

    Mat trans = Mat(4,4, CV_32F, translation);
    
    float alpha = 3.14/36;
    float val_x[] ={ 1, 0, 0, 0, 
        0, cos(alpha), -sin(alpha), 0,
        0, sin(alpha), cos(alpha), 0,
        0, 0, 0, 1}; 
    Mat rot_x = Mat(4,4, CV_32F, val_x );
    float beta = 0/5;
    float val_y[] ={ cos(beta), 0, sin(beta), 0, 
        0, 1, 0, 0,
        -sin(beta), 0, cos(beta), 0,
        0, 0, 0, 1}; 
    Mat rot_y = Mat(4,4, CV_32F, val_y );
    float gamma = 3.14/7.2;//3.14/6;
    float val_z[] ={ cos(gamma), -sin(gamma), 0, 0, 
        sin(gamma), cos(gamma), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1}; 
    Mat rot_z = Mat(4,4, CV_32F, val_z );
    Mat rot = rot_x*rot_z*rot_y;
    printf("meanx: %f\n", m_x);
    while (true) {
        usleep(50000);
        applyMat(trans.inv()*rot*trans, point_cloud);
        viz::WMesh mesh2 = viz::WMesh(point_cloud, triangle);
        trajectoryWindow.showWidget("point_cloudi2", mesh2);

        trajectoryWindow.spinOnce();
        break;
    }
    trajectoryWindow.spin();
    return 0;
}
