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
#include <sys/time.h>
#include <dirent.h>
#include "ins_nnie_interface.h"
#include "Tensor.h"
#include "util.h"
#include "opencv2/opencv.hpp"
#include "yolo_post.h"
#include <iostream>
#include <fstream>
/******************************************************************************
* function : show usage
******************************************************************************/
typedef unsigned char U_CHAR;


void yolov5DetectDemo(NNIE &yolov5, cv::Mat& src_img, cv::Mat & inference_img, std::string file_name)
{
    struct timeval tv1;
    struct timeval tv2;
    long t1, t2, time_run;

    gettimeofday(&tv1, NULL);

    yolov5.run(inference_img);

    gettimeofday(&tv2, NULL);
    t1 = tv2.tv_sec - tv1.tv_sec;
    t2 = tv2.tv_usec - tv1.tv_usec;
    time_run = (long)(t1 * 1000 + t2 / 1000);
    printf("inference time : %dms\n", time_run);

    Tensor output0 = yolov5.getOutputTensor(0);
    Tensor output1 = yolov5.getOutputTensor(1);
    Tensor output2 = yolov5.getOutputTensor(2);

    float src_img_w = (float)src_img.cols;
    float src_img_h = (float)src_img.rows;
    float inference_img_w = (float)inference_img.cols;
    float inference_img_h = (float)inference_img.rows;

    float conf_threshold = 0.25;
    float nms_threshold = 0.45;

    std::vector<Object> proposals;
    std::vector<size_t> picked;

    // const std::vector<std::vector<cv::Size2f>> anchors = {
    //     {{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}},
    //     {{30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}},
    //     {{10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}}};

    const std::vector<std::vector<cv::Size2f>> anchors = {
        {{42.0f, 22.0f}, {30.0f, 42.0f}, {68.0f, 48.0f}},
        {{13.0f, 14.0f}, {25.0f, 13.0f}, {21.0f, 25.0f}},
        {{5.0f, 4.0f}, {8.0f, 7.0f}, {17.0f, 7.0f}}};

    yolov5_generate_proposals(output0, anchors[2], 8, conf_threshold, proposals);

    yolov5_generate_proposals(output1, anchors[1], 16, conf_threshold, proposals);

    yolov5_generate_proposals(output2, anchors[0], 32, conf_threshold, proposals);
    
    qsort_descent_inplace(proposals);
    // for(size_t i = 0; i < proposals.size(); ++i) 
    //     printf("sort_confidence: %f\n", proposals[i].confidence);

    yolo_nms(proposals, picked, nms_threshold);
    std::vector<Object> objects(picked.size());
    float scale_w = src_img_w / inference_img_w;
    float scale_h = src_img_h / inference_img_h;

    std::fstream outfile(file_name, std::ios::out);
    printf("get %d objects\n", picked.size());
    for(size_t i = 0; i < picked.size(); ++i)
    {
        objects[i] = proposals[picked[i]];
        
        float x0 = objects[i].bbox.x * scale_w;
        float x1 = (objects[i].bbox.x + objects[i].bbox.width) * scale_w;
        float y0 = objects[i].bbox.y * scale_h;
        float y1 = (objects[i].bbox.y + objects[i].bbox.height) * scale_h;

        x0 = std::min(std::max(x0, 0.f), src_img_w - 1.0f);
        x1 = std::min(std::max(x1, 0.f), src_img_w - 1.0f);
        y0 = std::min(std::max(y0, 0.f), src_img_h - 1.0f);
        y1 = std::min(std::max(y1, 0.f), src_img_h - 1.0f);

        outfile << class_names[objects[i].class_label]  << " " << objects[i].confidence << " " << round(x0) << " " << round(y0) << " " << round(x1) << " " << round(y1) << std::endl;

        objects[i].bbox.x = x0;
        objects[i].bbox.y = y0;
        objects[i].bbox.width = x1 - x0;
        objects[i].bbox.height = y1 - y0;
    }
    outfile.close();
}

/******************************************************************************
* function 
******************************************************************************/
int main(int argc, char *argv[])
{
    //const char *model_path = "../models/inst_yolov5s_nofocus_decon.wk";
    const char *model_path = "../models/inst_yolov5s_nofocus_wlk_rgb_low_125pic.wk";
    //const char *model_path = "../models/inst_yolov5s_nofocus_wlk_rgb_high_pre_125pic.wk";
    
    const char *img_path = "./imglist.txt";

    std::string output_path = "lowpre_result_125pic/";

    std::ifstream infile(img_path, std::ios::in);
    if(!infile)
    {
        printf("error, can not open img path\n");
        return -1;
    }

    //char img_name[256];
    std::string img_name;
    NNIE yolov5;
    cv::Mat orig_img, img;
    printf("begin to initiate the model");
    yolov5.init(model_path);
    printf("yolov5 start\n");
    while(1)
    {
        std::getline(infile, img_name);
        if(!infile) return 0;
        orig_img = cv::imread(img_name);
        std::cout << "detect " << img_name << std::endl;
        if (orig_img.empty())
        {
            printf("read pic fail!\n");
            return 0;
        }
        resize(orig_img, img, cv::Size(416, 416));
        std::string file_name = img_name.substr(img_name.find_last_of("/") + 1, img_name.find_last_of(".") - img_name.find_last_of("/") - 1);
        file_name = output_path + file_name + ".txt";
        yolov5DetectDemo(yolov5, orig_img, img, file_name);
    }
   
    printf("yolov5 finish\n");
    return 0;
}