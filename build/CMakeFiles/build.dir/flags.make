# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# compile CUDA with /usr/local/cuda-12.2/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_DEFINES = 

CUDA_INCLUDES = -I/usr/lib/TensorRT/targets/x86_64-linux-gnu/samples -I/usr/lib/TensorRT/targets/x86_64-linux-gnu/samples/common -I/usr/local/cuda12.2/include -isystem=/usr/include/opencv4

CUDA_FLAGS =  --generate-code=arch=compute_61,code=[compute_61,sm_61] --generate-code=arch=compute_70,code=[compute_70,sm_70] --generate-code=arch=compute_75,code=[compute_75,sm_75] -std=c++14

CXX_DEFINES = 

CXX_INCLUDES = -I/usr/lib/TensorRT/targets/x86_64-linux-gnu/samples -I/usr/lib/TensorRT/targets/x86_64-linux-gnu/samples/common -I/usr/local/cuda12.2/include -isystem /usr/include/opencv4

CXX_FLAGS = -std=gnu++14

