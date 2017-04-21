#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stack>
using namespace std;
using namespace cv;

struct regionprop{
	vector<Point> pixel_list;
	int Area;
	int Perimeter;
	Point2i Centroid;
	Rect2i BoundingBox;
	Mat image;
};

class fc_detector{
public:
	string imgname;
	int edge_type;//1 stands for canny,0 for sobel
	bool add_edge;// 
	Mat image_in;
	Mat image_resize;
	Mat image_gray;
	Mat image_hist;
	Mat image_edge;
	Mat image_binary;
	Mat image_post;
	Mat image_resegment;
	Mat image_color;
	Size2i img_size;
	void imgResize(Mat&,Mat&,int);
	void imgResize(Mat&,Mat&);
	void imgHisteq(Mat&,Mat&);
	void imgEdge(Mat&,Mat&);
	void imadjust(const Mat1b& src, Mat1b& dst, int tol = 1, Vec2i in = Vec2i(0, 255), Vec2i out = Vec2i(0, 255));
	void imAdjust(const Mat1b& src, Mat1b& dst);
	void imgBinary(Mat&,Mat&);
	void imgPostprocess(Mat&,Mat&,Mat&);
	void imgResegment();
	void imgPainting();
	int icvprCcaBySeedFill(const Mat& , Mat& );//用区域生长法(8邻域)找连通域
	void icvprCcaByTwoPass(const Mat&, Mat& );  
	void icvprLabelColor(const Mat& , Mat& );   
	void bwareaopen(const Mat&, Mat&,vector<regionprop>,int);
	void imfillholes(const Mat&, Mat&);
	void getBaseProps(regionprop&,Size2i); 
	void getAllProps(regionprop&,Size2i); 
	Mat gaussian_kernal(int,double);
	vector<regionprop> getRegions(const Mat&,int);
	fc_detector(void);
	~fc_detector(void);
};