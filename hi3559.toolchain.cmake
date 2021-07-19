# set target machine operating system  eg. Linux  Windows Generic
set(CMAKE_SYSTEM_NAME Linux)

#set target machine architecture    
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER "aarch64-himix100-linux-gcc")
set(CMAKE_CXX_COMPILER "aarch64-himix100-linux-g++")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGARM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)

set(CMAKE_C_FLAGS "-march=armv8-a")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=armv8-a -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fpermissive -fexceptions")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")