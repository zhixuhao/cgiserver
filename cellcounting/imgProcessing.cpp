#include "cellDetector.h"

fc_detector::fc_detector(void)
{
};

fc_detector::~fc_detector(void)
{
};

void fc_detector::imgResize(Mat& imgin,Mat& imgout){
	pyrDown(imgin,imgout);
};

void fc_detector::imgResize(Mat& imgin,Mat& imgout,int s){
	double x = pow(2,s);
	Size dst_size = Size(int(imgin.cols/x),int(imgin.rows/x));
	pyrDown(imgin,imgout,dst_size);
};

void fc_detector::getBaseProps(regionprop &region,Size2i imgsize){
	int sum_row = 0;
	int sum_col = 0;
	int min_row = -1;
	int max_row = -1;
	int min_col = -1;
	int max_col = -1;
	for(int i=0; i < region.pixel_list.size();i++){
		int r = region.pixel_list[i].x;
		int c = region.pixel_list[i].y;
		sum_row += r;
		sum_col += c;
		if(r < min_row || min_row == -1){
			min_row = r;
		}
		if(c < min_col || min_col == -1){
			min_col = c;
		}
		if(r > max_row){
			max_row = r;
		}
		if(c > max_col){
			max_col = c;
		}
	}
	region.Area = region.pixel_list.size();
	int x = int(sum_row/region.pixel_list.size());
	int y = int(sum_col/region.pixel_list.size());
	x = (x < 0) ? 0:x;
	x = (x >= imgsize.width) ? imgsize.width:x;
	y = (y < 0) ? 0:y;
	y = (y >= imgsize.height) ? imgsize.height:y;
	region.Centroid = Point2i(x,y);
	region.pixel_list = region.pixel_list;
	region.BoundingBox = Rect2i(Point(min_col,min_row),Point(max_col+1,max_row+1));
}

void fc_detector::getAllProps(regionprop &region,Size2i imgsize){
	Mat img(imgsize,CV_8UC1,Scalar(1));
	int sum_row = 0;
	int sum_col = 0;
	int min_row = -1;
	int max_row = -1;
	int min_col = -1;
	int max_col = -1;
	for(int i=0; i < region.pixel_list.size();i++){
		int r = region.pixel_list[i].x;
		int c = region.pixel_list[i].y;
		sum_row += r;
		sum_col += c;
		if(r < min_row || min_row == -1){
			min_row = r;
		}
		if(c < min_col || min_col == -1){
			min_col = c;
		}
		if(r > max_row){
			max_row = r;
		}
		if(c > max_col){
			max_col = c;
		}
	}
	region.Area = region.pixel_list.size();
	int x = int(sum_row/region.pixel_list.size());
	int y = int(sum_col/region.pixel_list.size());
	x = (x < 0) ? 0:x;
	x = (x >= imgsize.width) ? imgsize.width:x;
	y = (y < 0) ? 0:y;
	y = (y >= imgsize.height) ? imgsize.height:y;
	region.Centroid = Point2i(x,y);
	region.pixel_list = region.pixel_list;
	region.BoundingBox = Rect2i(Point(min_col,min_row),Point(max_col+1,max_row+1));
	region.image = img(region.BoundingBox);
	Mat image_roi = region.image.clone();
	Mat kernel(3,3,CV_16SC1,Scalar(0));
	kernel.at<short>(0,1) = kernel.at<short>(1,0) = kernel.at<short>(1,2) = kernel.at<short>(2,1) = -1;
	kernel.at<short>(1,1) = 4;
	filter2D(region.image,image_roi,region.image.depth(),kernel,Point(-1,-1),0,BORDER_CONSTANT);
	int sum = 0;
	for(int i = 1;i < region.image.rows-1;i++){
		for(int j = 1; j < region.image.cols-1;j++){
			if(region.image.at<uchar>(i,j) == 2 && image_roi.at<uchar>(i,j) == 0){
				sum++;
			}
		}
	}
	region.Perimeter = region.Area - sum;
};

void fc_detector::imgHisteq(Mat& imgin,Mat& imgout){
	equalizeHist(imgin,imgout);
};


void fc_detector::imgEdge(Mat& imgin,Mat& imgout){
	GaussianBlur(imgin,imgin,Size(29,29),5,5);
	//cout << "GaussianBlur done" <<endl;
	Canny(imgin,imgout,30.*0.4,30);
};

Mat fc_detector::gaussian_kernal(int ksize,double sigma){
	Mat kernel(ksize,ksize,CV_32FC1);
	Point center_p(int((ksize+1)/2 - 1),int(int((ksize+1)/2 - 1)));
	float sum = 0;
	for(int i = 0;i < ksize;i++){
		for(int j = 0; j < ksize;j++){
			float val = float(exp(-(pow((i-center_p.x),2)+pow((j-center_p.y),2))/(2*pow(sigma,2))));
			kernel.at<float>(i,j) = val;
			sum += val;
		}
	}
	kernel = kernel.mul(1/sum);
	return kernel;
}


void fc_detector::imgBinary(Mat &imgin,Mat &imgout){
	imgout = imgin.clone();
	Mat1b img1,img2,img3,img33;
	Mat1b img4;
	Mat kernel_1 = gaussian_kernal(3,0.5);
	Mat kernel_2 = gaussian_kernal(59,45);
	filter2D(imgin,img1,imgin.depth(),kernel_1,Point(-1,-1),0,BORDER_REPLICATE);
	filter2D(imgin,img2,imgin.depth(),kernel_2,Point(-1,-1),0,BORDER_REPLICATE);
	img3 = img2 - img1;
 	imadjust(img3,img4);
	threshold(img4,imgout,0.012*255,255,0);
};

void fc_detector::imfillholes(const Mat& srcBw, Mat &dstBw)
{
    Size m_Size = srcBw.size();
    Mat Temp=Mat::zeros(m_Size.height+2,m_Size.width+2,srcBw.type());//ÑÓÕ¹Í¼Ïñ
    srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
 
    cv::floodFill(Temp, Point(0, 0), Scalar(255));
 
    Mat cutImg;//²Ã¼ôÑÓÕ¹µÄÍ¼Ïñ
    Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
 
    dstBw = srcBw | (~cutImg);
}

void fc_detector::imgPostprocess(Mat& imgbinary,Mat& imgedge,Mat& imgout){
/*	Mat imgedge_a;
	Mat kernel = getStructuringElement(0,Size(2,2));
	dilate(imgedge,imgedge_a,kernel);
	vector<vector<Point>> contours;
	vector<double> contour_areas;
	vector<Vec4i> hierarchy;
	findContours( imgbinary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
	for(int i = 0; i < contours.size();i++){
		contour_areas.push_back(contourArea(contours[i]));
	}
	cout<<contour_areas.size();*/
	Mat image_label,imgbinary_a;
	int num_region = icvprCcaBySeedFill(image_binary,image_label);
	vector<regionprop> regions = getRegions(image_label,num_region);
	bwareaopen(imgbinary,imgbinary_a,regions,300);
	Mat imgedge_a;
	dilate(imgedge,imgedge_a,getStructuringElement(0,Size(2,2)));
	Mat imgbinary_aa;
	dilate(imgbinary_a,imgbinary_aa,getStructuringElement(0,Size(15,15)));
	Mat imgedge_aa = imgedge_a.mul(imgbinary_aa);
	Mat img_bw_edge = imgedge_aa + imgbinary_a;
	regions.clear();
	num_region = icvprCcaBySeedFill(img_bw_edge,image_label);
	regions = getRegions(image_label,num_region);
	Mat img_bw_b;
	bwareaopen(img_bw_edge,img_bw_b,regions,100);
	Mat img_bw_bb;
	dilate(img_bw_b,img_bw_bb,getStructuringElement(0,Size(10,10)));
	erode(img_bw_bb,img_bw_bb,getStructuringElement(0,Size(10,10)));
	Mat img_reverse_b = ~img_bw_bb;
	imfillholes(img_reverse_b,imgout);
};


void fc_detector::imgResegment(){
};

void fc_detector::icvprCcaByTwoPass(const Mat& _binImg, Mat& _lableImg)  
{  
    // connected component analysis (4-component)  
    // use two-pass algorithm  
    // 1. first pass: label each foreground pixel with a label  
    // 2. second pass: visit each labeled pixel and merge neighbor labels  
    //   
    // foreground pixel: _binImg(x,y) = 1  
    // background pixel: _binImg(x,y) = 0  
  
  
    if (_binImg.empty() ||  
        _binImg.type() != CV_8UC1)  
    {  
        return;  
    }  
  
    // 1. first pass  
  
    _lableImg.release();  
    _binImg.convertTo(_lableImg, CV_32SC1);  
    _lableImg = _lableImg.mul(0.005);
	cout<<_lableImg;
    int label = 1;  // start by 2  
    vector<int> labelSet;  
    labelSet.push_back(0);   // background: 0  
    labelSet.push_back(1);   // foreground: 1  
  
    int rows = _binImg.rows - 1;  
    int cols = _binImg.cols - 1;  
    for (int i = 1; i < rows-1; i++)  
    {  
        int* data_preRow = _lableImg.ptr<int>(i-1);  
        int* data_curRow = _lableImg.ptr<int>(i);  
		int* data_nextRow = _lableImg.ptr<int>(i+1);  
        for (int j = 1; j < cols-1; j++)  
        {  
            if (data_curRow[j] == 1)  
            {  
                vector<int> neighborLabels;  
                neighborLabels.reserve(8);  
                int leftPixel = data_curRow[j-1];  
                int upPixel = data_preRow[j];  
				int leftupPixel = data_preRow[j-1];  
                int rightupPixel = data_preRow[j+1];  
				int leftdownPixel = data_nextRow[j-1];  
                int downPixel = data_nextRow[j];  
				int rightPixel = data_curRow[j+1];  
                int rightdownPixel = data_nextRow[j+1];  
                if ( leftPixel > 1)  
                {  
                    neighborLabels.push_back(leftPixel);  
                }  
                if (upPixel > 1)  
                {  
                    neighborLabels.push_back(upPixel);  
                }  
				if ( leftupPixel > 1)  
                {  
                    neighborLabels.push_back(leftupPixel);  
                }  
                if (rightupPixel > 1)  
                {  
                    neighborLabels.push_back(rightupPixel);  
                }
				if ( leftdownPixel > 1)  
                {  
                    neighborLabels.push_back(leftdownPixel);  
                }  
                if (downPixel > 1)  
                {  
                    neighborLabels.push_back(downPixel);  
                }  
				if ( rightPixel > 1)  
                {  
                    neighborLabels.push_back(rightPixel);  
                }  
                if (rightdownPixel > 1)  
                {  
                    neighborLabels.push_back(rightdownPixel);  
                }
                if (neighborLabels.empty())  
                {  
                    labelSet.push_back(++label);  // assign to a new label  
                    data_curRow[j] = label;  
                    labelSet[label] = label;  
                }  
                else  
                {  
                    sort(neighborLabels.begin(), neighborLabels.end());  
                    int smallestLabel = neighborLabels[0];    
                    data_curRow[j] = smallestLabel;  
  
                    // save equivalence  
                    for (size_t k = 1; k < neighborLabels.size(); k++)  
                    {  
                    /*    int tempLabel = neighborLabels[k];  
                        int& oldSmallestLabel = labelSet[tempLabel];  
                        if (oldSmallestLabel > smallestLabel)  
                        {                             
                            labelSet[oldSmallestLabel] = smallestLabel;  
                            oldSmallestLabel = smallestLabel;  
                        }                         
                        else if (oldSmallestLabel < smallestLabel)  
                        {  
                            labelSet[smallestLabel] = oldSmallestLabel;  
                        }*/  
						int tempLabel = neighborLabels[k];  
						if(labelSet[tempLabel] > smallestLabel){
							labelSet[tempLabel] = smallestLabel;
						}
                    }  
                }    
				cout<<"row:"<<i<<"col:"<<j<<endl<<endl;
				cout<<_lableImg<<endl<<"labelset:  ";
				for(int m = 0; m < labelSet.size();m++){
					cout<<labelSet[m];
				}
            }  
        }  
    }  
  
    // update equivalent labels  
    // assigned with the smallest label in each equivalent label set  
    for (size_t i = 2; i < labelSet.size(); i++)  
    {  
        int curLabel = labelSet[i];  
        int preLabel = labelSet[curLabel];  
        while (preLabel != curLabel)  
        {  
            curLabel = preLabel;  
            preLabel = labelSet[preLabel];  
        }  
        labelSet[i] = curLabel;  
    }  
  
  
    // 2. second pass  
    for (int i = 0; i < rows; i++)  
    {  
        int* data = _lableImg.ptr<int>(i);  
        for (int j = 0; j < cols; j++)  
        {  
            int& pixelLabel = data[j];  
            pixelLabel = labelSet[pixelLabel];   
        }  
    }  
}

int fc_detector::icvprCcaBySeedFill(const Mat& _binImg, Mat& _lableImg)  
{  
    // connected component analysis (4-component)  
    // use seed filling algorithm  
    // 1. begin with a foreground pixel and push its foreground neighbors into a stack;  
    // 2. pop the top pixel on the stack and label it with the same label until the stack is empty  
    //   
    // foreground pixel: _binImg(x,y) = 1  
    // background pixel: _binImg(x,y) = 0  
  
  
    if (_binImg.empty() ||  
        _binImg.type() != CV_8UC1)  
    {  
        return 0;  
    }  
  
    _lableImg.release();  
    _binImg.convertTo(_lableImg, CV_32SC1);  
    _lableImg = _lableImg.mul(0.005);//Ê¹µÃ255->1
    int label = 1;  // start by 2  
	Mat instack(_binImg.size(),CV_32SC1,Scalar(0));
    int rows = _binImg.rows - 1;  
    int cols = _binImg.cols - 1;  
    for (int i = 1; i < rows-1; i++)  
    {   
        for (int j = 1; j < cols-1; j++)  
        {  
            if (_lableImg.at<int>(i,j) == 1)  
            {  
                stack<pair<int,int> > neighborPixels;     
				if(instack.at<int>(i,j) == 0){
					neighborPixels.push(pair<int,int>(i,j));     // pixel position: <i,j>  
					instack.at<int>(i,j) = 1;
				}
                ++label;  // begin with a new label  
                while (!neighborPixels.empty())  
                {  
                    // get the top pixel on the stack and label it with the same label  
                    pair<int,int> curPixel = neighborPixels.top();  
                    int curR = curPixel.first;  
                    int curC = curPixel.second;  
                    _lableImg.at<int>(curR, curC) = label;  
  
                    // pop the top pixel  
                    neighborPixels.pop();  
					for(int k = 1;k <= 9; k++){
						if(k==5) continue;
						int offset_c = (k-1)%3 - 1;
						int offset_r = int((k-1)/3) - 1;
						int r = curR + offset_r;
						int c = curC + offset_c;
						if(r >= 0 && r <=rows && c >= 0 && c <= cols){
							if (_lableImg.at<int>(r, c) == 1){
								if(instack.at<int>(r,c) == 0){
									neighborPixels.push(pair<int,int>(r, c));  
									instack.at<int>(r,c) = 1;
								}
							}  
						}
					}
                }         
            }  
        }  
    }  
	return label-1;
}

vector<regionprop> fc_detector::getRegions(const Mat& img,int labels){
	 vector<regionprop> regions;
	 if (img.empty() || img.type() != CV_32SC1){  
        return regions;  
     }
	 regions.reserve(labels);
	 for(int i = 0; i < labels;i++){
		vector<Point> tmpvec;
		regionprop tmprp;
		tmpvec.reserve(1000);
		tmprp.pixel_list = tmpvec;
		regions.push_back(tmprp);
	 }
	 for(int i = 0; i < img.rows;i++){
		 for(int j = 0; j < img.cols;j++){
			int label = img.at<int>(i,j);
			if(label >= 2){
				Point p;
				p.x = j;
				p.y = i;
				regions[label-2].pixel_list.push_back(p);
			}
		 }
	 }
	 return regions;
}

void fc_detector::bwareaopen(const Mat& imgin, Mat& imgout,vector<regionprop> regions,int area){
	if(imgin.empty() || imgin.type() != CV_8UC1) return;
	imgout = imgin.clone();
	for(int i = 0; i < regions.size();i++){
		vector<Point> tmpvec = regions[i].pixel_list;
		if(tmpvec.size() < area){
			for(int j = 0; j < tmpvec.size();j++){
				imgout.at<uchar>(tmpvec[j].y,tmpvec[j].x) = 0;
			}
		}
	}
}

Scalar icvprGetRandomColor()  
{  
    uchar r = 255 * (rand()/(1.0 + RAND_MAX));  
    uchar g = 255 * (rand()/(1.0 + RAND_MAX));  
    uchar b = 255 * (rand()/(1.0 + RAND_MAX));  
    return Scalar(b,g,r);  
}  
  
  
void fc_detector::icvprLabelColor(const Mat& _labelImg, Mat& _colorLabelImg)   
{  
    if (_labelImg.empty() ||  
        _labelImg.type() != CV_32SC1)  
    {  
        return;  
    }  
  
    map<int, Scalar> colors;  
  
    int rows = _labelImg.rows;  
    int cols = _labelImg.cols;  
  
    _colorLabelImg.release();  
    _colorLabelImg.create(rows, cols, CV_8UC3);  
    _colorLabelImg = Scalar::all(0);  
  
    for (int i = 0; i < rows; i++)  
    {  
        const int* data_src = (int*)_labelImg.ptr<int>(i);  
        uchar* data_dst = _colorLabelImg.ptr<uchar>(i);  
        for (int j = 0; j < cols; j++)  
        {  
            int pixelValue = data_src[j];  
            if (pixelValue > 1)  
            {  
                if (colors.count(pixelValue) <= 0)  
                {  
                    colors[pixelValue] = icvprGetRandomColor();  
                }  
                Scalar color = colors[pixelValue];  
                *data_dst++   = color[0];  
                *data_dst++ = color[1];  
                *data_dst++ = color[2];  
            }  
            else  
            {  
                data_dst++;  
                data_dst++;  
                data_dst++;  
            }  
        }  
    }  
}  

void fc_detector::imgPainting(){
};

void vector_sort(vector<uchar> &vec,uchar insert_val){
	int s = vec.size();
	int count = s;
	while(true){
		if(count <= 0){
			vec.insert(vec.begin(),insert_val);
			break;
		}
		else{
			if(insert_val <= vec[count-1]){
				vec.insert(vec.begin()+count,insert_val);
				break;
			}
		}
		count--;
	}
}

void fc_detector::imAdjust(const Mat1b& src, Mat1b& dst){
	dst = src.clone();
	vector<uchar> vec;
	int rows = src.rows;
	int cols = src.cols;
	int total_pixels = int(rows*cols*0.01);
	int count = 0;
	for(int i = 0;i < rows;i++){
		for(int j = 0;j < cols;j++){
			uchar val = src(i,j);
			if(val > 0){
				if(count < total_pixels){
					vector_sort(vec,val);
					count++;
				}
				else{
					if(val > vec[total_pixels-1]){
						vec.pop_back();
						vector_sort(vec,val);
					}
				}
			}
		}
	}
	int max_val = vec[total_pixels-1];
	float t = 255.0/max_val;
	//cout<<max_val<<endl;
	//cout<<t<<endl;
	for(int i = 0;i < rows;i++){
		for(int j = 0;j < cols;j++){
			uchar val = uchar(src(i,j)*t);
			int x = src(i,j)*t;
			if(x > 255){val = 255;}
			dst(i,j) = val;
		}
	}
}



void fc_detector::imadjust(const Mat1b& src, Mat1b& dst, int tol, Vec2i in, Vec2i out){
    // src : input CV_8UC1 image
    // dst : output CV_8UC1 imge
    // tol : tolerance, from 0 to 100.
    // in  : src image bounds
    // out : dst image buonds

    dst = src.clone();
    tol = max(0, min(100, tol));

    if (tol > 0)
    {
        // Compute in and out limits

        // Histogram
        vector<int> hist(256, 0);
        for (int r = 0; r < src.rows; ++r) {
            for (int c = 0; c < src.cols; ++c) {
                hist[src(r,c)]++;
            }
        }

        // Cumulative histogram
        vector<int> cum = hist;
        for (int i = 1; i < hist.size(); ++i) {
            cum[i] = cum[i - 1] + hist[i];
        }

        // Compute bounds
        int total = src.rows * src.cols;
        int low_bound = total * tol / 100;
        int upp_bound = total * (100-tol) / 100;
        in[0] = distance(cum.begin(), lower_bound(cum.begin(), cum.end(), low_bound));
        in[1] = distance(cum.begin(), lower_bound(cum.begin(), cum.end(), upp_bound));

    }

    // Stretching
    float scale = float(out[1] - out[0]) / float(in[1] - in[0]);
    for (int r = 0; r < dst.rows; ++r)
    {
        for (int c = 0; c < dst.cols; ++c)
        {
            int vs = max(src(r, c) - in[0], 0);
            int vd = min(int(vs * scale + 0.5f) + out[0], out[1]);
            dst(r, c) = saturate_cast<uchar>(vd);
        }
    }
}