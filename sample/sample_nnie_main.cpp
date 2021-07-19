#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <vector>
#include <sys/ioctl.h>
#include <dirent.h>
#include <string>
#include "ins_nnie_interface.h"
#include "Tensor.h"
#include "util.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"
#include <time.h>
typedef unsigned char U_CHAR;
#define MAX_CLASS_COUNT 6
 NNIE yolov3_mnas[MAX_CLASS_COUNT];

 /*
 模型类表
 0:刀具
 1：抽烟
 2：餐盘
 3：穿衣戴帽-衣服
 4：穿衣戴帽-帽子
 5: 垃圾桶
 */
            
 const std::string model_file_list[MAX_CLASS_COUNT]={"/root/models/yolov3-knife.wk",
                                                     "/root/models/yolov3-cig.wk",
                                                     "/root/models/yolov3-dish.wk",
                                                     "/root/models/yolov3-whitecloth.wk",
                                                     "/root/models/yolov3-whitehat.wk",
                                                     "/root/models/yolov3-trashbin.wk"};

int model_class_count[MAX_CLASS_COUNT]={1,2,1,2,2,2};
const std::string class_name_list[][4]={ 
	            {"backgroud","knife"},
                {"backgroud","person","cigarette"},
                {"backgroud","dish"},
                {"backgroud","regular","irregular"},
                {"backgroud","regular","irregular"},
                {"backgroud","open","close"}
};



 int last_class_type[MAX_CLASS_COUNT]={-1,-1,-1,-1,-1,-1};
 
 char gresult[4096];

extern "C" int  nnie_yolov3_init(int class_type)
{
	printf("build time:%s,%s\n",__DATE__,__TIME__);

	if (last_class_type[class_type]!=class_type)
		{
		last_class_type[class_type]=class_type;
	    yolov3_mnas[class_type].init(model_file_list[class_type].c_str());	
		}
	
	return 0;
}
extern "C" char * nnie_yolov3_detect(int class_type, const char *image_path,float conf_threshold)
{
	// printf("yolov3 start:%s\n",image_path);
	

	  char outname[200];
	  sprintf(outname, "/tmp/temp_%d.bgr",class_type);
    cv::Mat orig_img, img;
    orig_img = cv::imread(image_path);
    if(orig_img.empty())
    {
    	
	    sprintf(gresult,"{\"count\":%d,\"data\":[]}",0);
      printf("%s",gresult);
      return gresult;
    }
    resize(orig_img, img, cv::Size(416, 416));
	  U_CHAR *data = (U_CHAR*)img.data;
	  int step = img.step;
	 // printf("Step: %d, height: %d, width: %d\n",step, img.rows, img.cols);

  	FILE *fp = fopen(outname, "wb");
  	int h = img.rows;
  	int w = img.cols;
  	int c = img.channels();

  	for (int k = 0; k<c; k++) {
  		for (int i = 0; i<h; i++) {
  			for (int j = 0; j<w; j++) {
  				fwrite(&data[i*step + j*c + k], sizeof(U_CHAR), 1, fp);
  			}
  		}
  	}
    fflush(fp);
  	fclose(fp);

    struct timeval tv1;
    struct timeval tv2;
    long t1, t2, time_;

    gettimeofday(&tv1, NULL);
    yolov3_mnas[class_type].run(outname);

    gettimeofday(&tv2, NULL);
    t1 = tv2.tv_sec - tv1.tv_sec;
    t2 = tv2.tv_usec - tv1.tv_usec;
    time_ = (long)(t1 * 1000 + t2 / 1000);
    printf("yolov3 NNIE inference time : %dms\n", time_);

    gettimeofday(&tv1, NULL);

    Tensor output0 = yolov3_mnas[class_type].getOutputTensor(0);
    Tensor output1 = yolov3_mnas[class_type].getOutputTensor(1);
    Tensor output2 = yolov3_mnas[class_type].getOutputTensor(2);

    /*yolov3的参数*/
    int img_width = orig_img.cols;;
    int img_height = orig_img.rows;
    int num_classes = model_class_count[class_type];
    int kBoxPerCell = 3;

    int feature_index0 = 0;
    int feature_index1 = 1;
    int feature_index2 = 2;

  //  float conf_threshold = 0.5;
    float nms_threshold = 0.5;
    int is_nms = 1;

    std::vector<int> ids;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    const std::vector<std::vector<cv::Size2f>> anchors = {
        {{116, 90}, {156, 198}, {373, 326}},
        {{30, 61}, {62, 45}, {59, 119}},
        {{10, 13}, {16, 30}, {33, 23}}};

    parseYolov3Feature(img_width,
                       img_height,
                       num_classes,
                       kBoxPerCell,
                       feature_index0,
                       conf_threshold,
                       anchors[0],
                       output0,
                       ids,
                       boxes,
                       confidences);

    parseYolov3Feature(img_width,
                       img_height,
                       num_classes,
                       kBoxPerCell,
                       feature_index1,
                       conf_threshold,
                       anchors[1],
                       output1,
                       ids,
                       boxes,
                       confidences);

    parseYolov3Feature(img_width,
                       img_height,
                       num_classes,
                       kBoxPerCell,
                       feature_index2,
                       conf_threshold,
                       anchors[2],
                       output2,
                       ids,
                       boxes,
                       confidences);

    std::vector<int> indices;
    /*print result, this sample has 81 classes:
      class 0:background      class 1:person       class 2:bicycle         class 3:car            class 4:motorbike      class 5:aeroplane
      class 6:bus             class 7:train        class 8:truck           class 9:boat           class 10:traffic light
      class 11:fire hydrant   class 12:stop sign   class 13:parking meter  class 14:bench         class 15:bird
      class 16:cat            class 17:dog         class 18:horse          class 19:sheep         class 20:cow
      class 21:elephant       class 22:bear        class 23:zebra          class 24:giraffe       class 25:backpack
      class 26:umbrella       class 27:handbag     class 28:tie            class 29:suitcase      class 30:frisbee
      class 31:skis           class 32:snowboard   class 33:sports ball    class 34:kite          class 35:baseball bat
      class 36:baseball glove class 37:skateboard  class 38:surfboard      class 39:tennis racket class 40bottle
      class 41:wine glass     class 42:cup         class 43:fork           class 44:knife         class 45:spoon
      class 46:bowl           class 47:banana      class 48:apple          class 49:sandwich      class 50orange
      class 51:broccoli       class 52:carrot      class 53:hot dog        class 54:pizza         class 55:donut
      class 56:cake           class 57:chair       class 58:sofa           class 59:pottedplant   class 60bed
      class 61:diningtable    class 62:toilet      class 63:vmonitor       class 64:laptop        class 65:mouse
      class 66:remote         class 67:keyboard    class 68:cell phone     class 69:microwave     class 70:oven
      class 71:toaster        class 72:sink        class 73:refrigerator   class 74:book          class 75:clock
      class 76:vase           class 77:scissors    class 78:teddy bear     class 79:hair drier    class 80:toothbrush*/
      
   

    std::vector<ObjectDetection> detection_results;

    if (is_nms)
    {
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    }
    else
    {
        for (int i = 0; i < boxes.size(); ++i)
        {
            indices.push_back(i);
        }
    }
  
   
	//{"data":["class":"knife","score":,0,4,"x":1,"y":2,"w":1,"h":2],[]}
	char temp[4096]={0};
	sprintf(temp,"{\"count\":%d,\"data\":[",indices.size());
	 std::string r=temp;
   int irregular=0;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];

        // remap box in src input size.
        auto remap_box = RemapBoxOnSrc(cv::Rect2d(box), img_width, img_height);
        ObjectDetection object_detection;
        object_detection.box = remap_box;
        object_detection.cls_id = ids[idx] + 1;
        object_detection.confidence = confidences[idx];
        detection_results.push_back(std::move(object_detection));

        float xmin = object_detection.box.x;
        float ymin = object_detection.box.y;
        float xmax = object_detection.box.x + object_detection.box.width;
        float ymax = object_detection.box.y + object_detection.box.height;
        float confidence = object_detection.confidence;
        int cls_id = object_detection.cls_id;
		
        const char *cls_name = class_name_list[class_type][cls_id].c_str();
       // printf("%d %s %.3f %.3f %.3f %.3f %.3f\n", cls_id, cls_name, confidence, xmin, ymin, xmax, ymax);
    		memset(temp,0,sizeof(temp));
    		sprintf(temp,"{\"class\":\"%s\",\"score\":%.5f,\"x\":%d,\"y\":%d,\"w\":%d,\"h\":%d}",cls_name,confidence,(int)xmin,(int)ymin,(int)(xmax-xmin),(int)(ymax-ymin));
    		r+=temp;
    		if (i!= indices.size()-1)
    		{
    			r+=",";
    		}
        cv::rectangle(orig_img, cv::Point(xmin,ymin),cv::Point(xmax,ymax),cv::Scalar(255,0,0), 3);
        
        /*
        std::string class_name;
        class_name=cls_name;
        if (class_name=="irregular")
            irregular++;
        */
    }
  	r+="]}";
  	memset(gresult,0,sizeof(gresult));
  	strcpy(gresult,r.c_str());
   
   
   /*
    if(irregular >= 2)
    {
   	  char save_path[50];
      time_t timep;
      time (&timep);
      sprintf(save_path, "/tmp/d_%d.jpg",time(NULL));
      cv::imwrite(save_path,orig_img);
    }
    */


    gettimeofday(&tv2, NULL);
    t1 = tv2.tv_sec - tv1.tv_sec;
    t2 = tv2.tv_usec - tv1.tv_usec;
    time_ = (long)(t1 * 1000 + t2 / 1000);
	return gresult;
}	


/******************************************************************************
* function 
******************************************************************************/
int main(int argc, char *argv[])
{
    const char *model_path_index = argv[1];
    const char *image_path = argv[2];
    long int times=0;
  	printf("build time:%s,%s\n",__DATE__,__TIME__);
  	int mindex=atoi(model_path_index);
      nnie_yolov3_init(mindex);
      unsigned int c=0;
      char *r;
   	float conf_threshold=0.6;
    r=nnie_yolov3_detect(mindex,image_path,conf_threshold);
  	printf("Result:=%s\n",r);
    return 0;
}