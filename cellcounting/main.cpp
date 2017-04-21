#include "cellDetector.h"
#include <string>
#include <sstream>
#include <unistd.h>  
#include <dirent.h>  
#include <stdlib.h>  
#include <stdio.h>

vector<string> getFiles(string cate_dir)  
{  
    vector<string> files;
    DIR *dir;  
    struct dirent *ptr;  
    char base[1000];  
   
    if ((dir=opendir(cate_dir.c_str())) == NULL){  
        cout << "Open dir error..." <<endl;  
  		exit(1);
    }  
    
    while ((ptr=readdir(dir)) != NULL){  
        if(strcmp(ptr->d_name,".") == 0 || strcmp(ptr->d_name,"..") == 0)
            continue;  
        else if(ptr->d_type == 8)    ///file  
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);  
            files.push_back(ptr->d_name);  
        else if(ptr->d_type == 10)    ///link file  
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);  
            continue;  
        else if(ptr->d_type == 4)    ///dir  
        {  
            //files.push_back(ptr->d_name);  
            /* 
                memset(base,'\0',sizeof(base)); 
                strcpy(base,basePath); 
                strcat(base,"/"); 
                strcat(base,ptr->d_nSame); 
                readFileList(base); 
            */  
            continue;
        }  
    }  
    closedir(dir);   
    //sort(files.begin(), files.end());  
    return files;  
}  

void img_process(fc_detector fc, string filename, string dstcolor, string dstlabel){
	//cout << filename << endl;
	fc.image_in = imread(filename);
	if(fc.image_in.rows > 2000){
		pyrDown(fc.image_in,fc.image_in);
	}
	//cout << fc.image_in.rows << fc.image_in.cols << endl;
	fc.img_size = Size2i(fc.image_in.cols,fc.image_in.rows);
	cvtColor(fc.image_in,fc.image_gray,CV_RGB2GRAY);
	fc.imgHisteq(fc.image_gray,fc.image_hist);
	fc.imgEdge(fc.image_hist,fc.image_edge);
	fc.imgBinary(fc.image_gray,fc.image_binary);
	Mat imgbinary_a;
	fc.imgPostprocess(fc.image_binary,fc.image_edge,imgbinary_a);
	Mat img_label,img_color;
	int num_labels = fc.icvprCcaBySeedFill(imgbinary_a,img_label);
	vector<regionprop> regions = fc.getRegions(img_label,num_labels);
	for(int i = 0; i < regions.size(); i++){
		stringstream stream;
		stream << i+1;
		string strnum;
		stream >> strnum;
		fc.getBaseProps(regions[i],fc.img_size);
		putText(fc.image_in,strnum,regions[i].Centroid,CV_FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255));
	}
	fc.icvprLabelColor(img_label,img_color);
	imwrite(dstlabel,fc.image_in);
	imwrite(dstcolor,img_color);
}

void outhead(){
	cout << "Content-type:text/html\r\n\r\n";
	//cout << "<html>\n";
	//cout << "<body>";
}

void outfoot(){
	//cout <<"</body></html>";
}

void outimg(string path, string colorimg, string labelimg){
	cout << "<div class = 'img_container'>";
	cout << "<img src = '" + path + colorimg + "'>";
	cout << "<img src = '" + path + labelimg + "'>";
	cout << "</div>";
}

void outhtml(){

}

/*int main(int argc,char *argv[]){
	char basePath[100];  
    memset(basePath, '\0', sizeof(basePath));  
    getcwd(basePath, 999); 
    string curpath(basePath);
	string srcpath = curpath + "/file/";
	string dstpath = curpath + "/results/";
	string srcpath = curpath + "/file/";
	string dstpath = curpath + "/results/";
	//cout << srcpath << endl;
	vector<string> files = getFiles(srcpath);
	string prefix(argv[1]);
	//cout << "prefix : " << prefix << endl;
	fc_detector fc;
	outhead();
	for(int i = 0; i < files.size(); i ++){
		string filename = files[i];
		//cout << filename << endl;
		if(filename.find(prefix) != string::npos){
			string dstname = "dst" + filename.substr(filename.find(prefix));
			img_process(fc,srcpath + filename,dstpath + dstname);
			outimg("",dstname);
		}
	}
	outfoot();
	return 0;
}*/

int main(){
	string srcpath = "../bioinf/CellCounting/file/";
	string dstpath = "../bioinf/CellCounting/results/";
	string showpath = "results/";
	vector<string> files = getFiles(srcpath);
	string prefix(getenv("QUERY_STRING"));
	//cout << "prefix : " << prefix << endl;
	fc_detector fc;
	outhead();
	for(int i = 0; i < files.size(); i ++){
		string filename = files[i];
		//cout << filename << endl;
		if(filename.find(prefix) != string::npos){
			string dstcolor = "color" + filename.substr(filename.find(prefix)) + ".png";
			string dstlabel = "label" + filename.substr(filename.find(prefix)) + ".png";
			img_process(fc,srcpath + filename, dstpath + dstcolor, dstpath + dstlabel);
			outimg(showpath,dstcolor,dstlabel);
		}
	}
	outfoot();
	return 0;
}

