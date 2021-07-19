cmake_minimum_required(VERSION 2.8)

project(yolo_nnie)
#-fno-rtti -fno-exceptions
set(CMAKE_CXX_FLAGS "-std=c++11 -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math  -fpermissive")

set(OpenCV_DIR "third_party/opencv3/share/OpenCV/")
find_package(OpenCV REQUIRED)
message(${OpenCV_VERSION})

include_directories(./src)
include_directories(./third_party/hisi/include)

#set(mylibs mpi aacdec aacenc detail_ap dnvqe dpu_match dpu_rect dsp hdmi hdr_ap hi_cipher hiavslut hifisheyecalibrate ive md mfnr_ap mpi_photo pciv
#pos_query securec sfnr_ap svpruntime tde upvqe VoiceEngine nnie)

#find_library(COMM_LIB ${mylibs} ./third_party/hisi/lib NO_DEFAULT_PATH)
#message(${COMM_LIB})
set(LIB_DIR ${PROJECT_SOURCE_DIR}/third_party/hisi/lib)
link_directories(${LIB_DIR})

link_libraries(${COMM_LIB})
aux_source_directory(./src/ all_source)

add_executable(yolo_nnie ./sample/yolov5-nnie-sample.cpp ${all_source})

target_link_libraries(yolo_nnie ${OpenCV_LIBS} pthread dl m mpi aacdec aacenc detail_ap dnvqe dpu_match dpu_rect dsp hdmi hdr_ap hi_cipher hiavslut hifisheyecalibrate ive md mfnr_ap mpi_photo pciv
pos_query securec sfnr_ap svpruntime tde upvqe VoiceEngine nnie ) 