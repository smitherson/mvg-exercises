#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;



Mat pointToMat(Point2f point) {
    float arr[] = {point.x, point.y, 1};
    cv::Mat mat = Mat(3, 1, CV_32F, arr);
    return mat.clone();
}
//   A = [0 -v(3) v(2) ; v(3) 0 -v(1) ; -v(2) v(1) 0];

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

void printType(Mat m) {
    cout <<type2str(m.type()).c_str() << endl;
}

void drawPoints(Mat img, vector<Point2f> points, Mat& res) {
    res = img.clone();
    for (int i=0;i< points.size();i++) {
        circle( res,
         points[i],
         2.0,
         Scalar( 0, 0, 255 ));
    }
}

void printVector(vector<Point2f> v) {

    for(int i=0;i<v.size();++i) {
        printf("%.2f %.2f ", v[i].x, v[i].y);
    }
    printf("\n");
}
void printVector(vector<float> v) {

    for(int i=0;i<v.size();++i) {
        printf("%.2f ", v[i]);
    }
    printf("\n");
}



//works only for vectors
Mat kron( Mat u, Mat v) {
    Mat res(u.rows*v.rows, 1, CV_32F);
    int row = 0;
    for (int i=0; i<u.rows; i++)
    {
        for (int j=0; j<v.rows; j++)
        {
            res.at<float>(row++,0) = u.at<float>(i, 0) * v.at<float>(j, 0);
        }
    }
    return res.clone();
}

void printMat(Mat m) {
    cout << m.rows << "x" << m.cols << " " << type2str(m.type()).c_str() << endl;
    cout << m << endl << endl;
}

Mat transpose(Mat m) {
    Mat mt;
    transpose(m, mt);
    return mt.clone();
}


Mat se3Exp(Mat_<float> twist) {
    float r[] = {twist(0,3), twist(0,4), twist(0, 5)};
    
    Mat_<float> R; 
    Rodrigues(Mat(3,1,CV_32F, r), R);

    float M[] = {0, 0, 0, twist(0,0),
                 0, 0, 0, twist(0,1),
                 0, 0, 0, twist(0,2),
                 0, 0, 0, 1};
    
    Mat_<float> T = Mat(4,4,CV_32F, M); 
    R.copyTo(T(cv::Rect(0,0,3,3)));
    return T.clone();
}

Mat se3Log(Mat_<float> T) {
    
    Mat_<float> twist = Mat::zeros(1,6, CV_32F);
    twist(0,0) = T(0,3);
    twist(0,1) = T(1,3);
    twist(0,2) = T(2,3);

    Mat_<float> R; 
    Rodrigues(T(Rect(0,0,3,3)), R);
    twist(0,3) = R(0,0);
    twist(0,4) = R(1,0);
    twist(0,5) = R(2,0);

    return twist.clone();
}

void downscaleK(Mat_<float> &K, int num) {
    if(num<=1) {
        return;
    }
    
    // this is because we interpolate in such a way, that 
    // the image is discretized at the exact pixel-values (e.g. 3,7), and
    // not at the center of each pixel (e.g. 3.5, 7.5).
    K(0,0) =  (float)K(0,0)/2.0;
    K(0,1) = 0;
    K(0,2) = (float)(K(0,2)+0.5)/2.0-0.5;

    K(1,0) = 0;
    K(1,1) = (float)K(1,1)/2.0;
    K(1,2) = (float)(K(1,2)+0.5)/2.0-0.5;
    downscaleK(K, num-1);
}

template<typename T>
void downscaleImage(Mat &image, int num) {
    if(num<=1) {
        return;
    }
    Mat smaller_image = Mat(image.rows/2,image.cols/2, CV_32F);

    for (int i=0; i<smaller_image.rows; i++) {
        for(int j=0; j<smaller_image.cols; j++) {
           smaller_image.at<T>(i,j) = (T) (( image.at<T>(2*i,2*j) + 
                                          image.at<T>(2*i+1,2*j) + 
                                          image.at<T>(2*i,2*j+1) +
                                          image.at<T>(2*i+1,2*j+1))*0.25);
        }
    }   
 
    image = smaller_image.clone();
    downscaleImage<T>(image, num -1 );
}
template<typename T>
void upscaleImage(Mat &image, int num) {
    if(num<=1) {
        return;
    }
    Mat bigger_image = Mat_<T>(image.rows*2,image.cols*2);

    for (int i=0; i<bigger_image.rows; i++) {
        for(int j=0; j<bigger_image.cols; j++) {
           bigger_image.at<T>(i,j) = (T) image.at<T>(i/2,j/2);
        }
    }   
 
    image = bigger_image.clone();
    upscaleImage<T>(image, num -1 );
}

template<typename T>
void downscaleDepth(Mat& image, int num){

    if(num<=1) {
        return;
    }

    Mat smaller = Mat_<T>(image.rows/2,image.cols/2);
    for (int i=0; i<smaller.rows; i++) {
        for(int j=0; j<smaller.cols; j++) {
            int denominator =  (int)(image.at<T>(2*i,2*j) > 0)  +
                              (int)(image.at<T>(2*i+1,2*j) > 0)  +
                              (int)(image.at<T>(2*i,2*j+1) > 0)  +
                              (int)(image.at<T>(2*i+1,2*j+1) > 0);

 
            if (denominator > 0) {
                smaller.at<T>(i,j) = (T)(( image.at<T>(2*i,2*j) + 
                                          image.at<T>(2*i+1,2*j) + 
                                          image.at<T>(2*i,2*j+1) +
                                          image.at<T>(2*i+1,2*j+1))/denominator);
            } else {
                 smaller.at<T>(i,j) = 0;
            }
        }
    }   
    image = smaller.clone();
    downscaleDepth<T>( image, num -1 );
}

template<typename T>
void calculateGradients(Mat_<T> image, Mat_<T>& Idrow, Mat_<T>& Idcol) {

/*    Mat Ix_temp, Iy_temp;
        
    Idrow = Mat_<T>(image.rows, image.cols);
    Idcol = Mat_<T>(image.rows, image.cols);
 
    Sobel( image, Ix_temp, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT );
    Sobel( image, Iy_temp, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT );
    //convertScaleAbs(Ix_temp, Ix);
    //convertScaleAbs(Iy_temp, Iy);
    Idrow = Ix_temp;
    Idcol = Iy_temp;*/

    Idrow = Mat_<T>(image.rows, image.cols,(T)0);
    Idcol = Mat_<T>(image.rows, image.cols, (T)0);
    //Ixy = Mat_<T>(image.rows, image.cols);

    for (int row=1; row<image.rows-1; row++) {
        for(int col=0; col<image.cols; col++) {
            Idrow(row,col) = (image(row+1,col)-image(row-1,col)) /2 ;
        }
    }
    for (int row=0; row<image.rows; row++) {
        for(int col=1; col<image.cols; col++) {
            Idcol(row,col) = (image(row,col+1)-image(row,col-1)) /2 ;
        }
    }

}

template<typename T>
cv::Mat_<T> hat(Mat_<T> vec) {
    T skew[] = {
        0,                  -vec(2,0), vec(1,0),
        vec(2,0), 0,                  -vec(0,0),
       -vec(1,0), vec(0,0),  0};

    cv::Mat_<T> skewMat = Mat_<T>(3, 3, skew);
    return skewMat.clone();
}

cv::Point3f MPoint3fMult(cv::Mat M, const cv::Point3f& p)
{ 
    cv::Mat_<float> src(4/*rows*/,1 /* cols */); 

    src(0,0)=p.x; 
    src(1,0)=p.y; 
    src(2,0)=p.z; 
    src(3,0)=1.0; 

    cv::Mat_<float> dst = M*src; //USE MATRIX ALGEBRA 
    return cv::Point3f(dst(0,0), dst(1,0), dst(2,0)); 
} 
