# yolov5-nnie
yolov5s nnie

YOLOv5 pytorch -> onnx -> caffe -> .wk
1、模型是yolov5s,将focus层替换成了stride为2的conv层。reshape和permute层也做了调整。具体的修改过程可以参考这个大佬的文章：https://blog.csdn.net/tangshopping/article/details/110038605

2、模型是在hi3559av100上跑的，mapper版本是1.2.

3、用法：
```
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../hi3559.toolchain.cmake ..
make -j4
./yolo_nnie
```

**reference**
> https://blog.csdn.net/tangshopping/article/details/110038605  
> watermelooon/nnie_yolo  
> https://github.com/ultralytics/yolov5  
> https://github.com/Wulingtian/yolov5_caffe  
> https://github.com/Wulingtian/yolov5_onnx2caffe  
