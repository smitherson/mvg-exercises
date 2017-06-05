#include <iostream>
#include "opencv2/core.hpp"

using namespace cv;
using namespace std;

int main()
{
    cout << "Built with OpenCV " << CV_VERSION << endl;
    float val[] = {2,6,7,8,5,
                 6,9,6,8,5,
                 7,6,1,7,5,
                 8,8,7,12,5,
                 5,5,5,5,5};
    float val2[] = {2,6,7,8,5,
                 6,9,6,8,5,
                 7,6,1,7,5,
                 8,8,7,12,5,
                 5,5,5,5,0};
    float val3[] = {1,2,3,4,5};

    Mat mat1 = Mat(5,5, CV_32F, val);
    Mat mat2 = Mat(5,5, CV_32F, val2);
    Mat b = Mat(5,1, CV_32F, val3);
//  cout << mat1 << endl;
  cout << mat1.inv()*b << endl;
//  cout << mat2 << endl;
  cout << mat2.inv(DECOMP_SVD)*b << endl;

//  Mat eigenVal, eigenVec;

//  eigen(mat1, eigenVal, eigenVec);

//  cout << eigenVal << eigenVec;
//  cout << "--------\n";
//  eigen(mat2, eigenVal, eigenVec);

//  cout << eigenVal << eigenVec;

//    SVD svd = SVD(mat1);
   
    //icout << Mat::diag(svd.w) << endl; 
//    cout << mat1 << endl;
//    cout << svd.u*Mat::diag(svd.w)*svd.vt << endl;
   
    return 0;

