#include "stdio.h"
#include "ins_nnie_interface.h"
#include "nnie_core.h"
#include "Tensor.h"
#include "util.h"


NNIE::NNIE()
{
}
NNIE::~NNIE()
{
    NNIE_Param_Deinit(&s_stNnieParam_, &s_stModel_);
}
void NNIE::init(const char *model_path, const int image_height = 416, const int image_width = 416)
{
    model_path_ = model_path;
    image_height_ = image_height;
    image_width_ = image_width;
    load_model(model_path, &s_stModel_);
    nnie_param_init(&s_stModel_, &stNnieCfg_, &s_stNnieParam_);
}

void NNIE::run(const char *file_path)
{

    int file_length = 0;
    FILE *fp = fopen(file_path, "rb");
    if (fp == NULL)
    {
        printf("open %s failed\n", file_path);
        return;
    }

    fseek(fp, 0L, SEEK_END);
    file_length = ftell(fp);
    fseek(fp, 0L, SEEK_SET);

    unsigned char *data = (unsigned char *)malloc(sizeof(unsigned char) * file_length);

    fread(data, file_length, 1, fp);

    fclose(fp);

    NNIE_Forward_From_Data(data, &s_stModel_, &s_stNnieParam_, output_tensors_);

    free(data);
}

void NNIE::run(const unsigned char *data)
{
    NNIE_Forward_From_Data(data, &s_stModel_, &s_stNnieParam_, output_tensors_);
}

void NNIE::run(cv::Mat& img)
{
    int step = img.step;
    unsigned char* src_data = (unsigned char*)img.data;
    unsigned char* dst_data = (unsigned char*)malloc(sizeof(unsigned char) * step * img.cols);
    int h = img.rows;
    int w = img.cols;
    int c = img.channels();
    int count = 0;
	  for (int k = 0; k<c; k++) {
	  	for (int i = 0; i<h; i++) {
	  		for (int j = 0; j<w; j++) {
	  			dst_data[count++] = src_data[i*step + j*c + k];
	  		}
	  	}
	  }

    NNIE_Forward_From_Data(dst_data, &s_stModel_, &s_stNnieParam_, output_tensors_);

    free(dst_data);
}

void NNIE::finish()
{
    NNIE_Param_Deinit(&s_stNnieParam_, &s_stModel_);
}

Tensor NNIE::getOutputTensor(int index)
{

    return output_tensors_[index];
}
void savebgr(cv::Mat& img, const char *outname)
{
    int step = img.step;
    unsigned char* data = (unsigned char*)img.data;
    FILE *fp = fopen(outname, "wb");
    int h = img.rows;
    int w = img.cols;
    int c = img.channels();

	  for (int k = 0; k<c; k++) {
	  	for (int i = 0; i<h; i++) {
	  		for (int j = 0; j<w; j++) {
	  			fwrite(&data[i*step + j*c + k], sizeof(unsigned char), 1, fp);
	  		}
	  	}
	  }
    fclose(fp);
}