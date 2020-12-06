NVCC := clang_cf++
NVCC_FLAGS := -O3 -std=c++11 --cuda-gpu-arch=sm_30
INC := -I../common
LIBS := -lcudart_static -lpthread -lm -ldl -lrt -lstdc++
