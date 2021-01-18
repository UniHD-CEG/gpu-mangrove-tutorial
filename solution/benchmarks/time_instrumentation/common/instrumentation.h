#ifndef INSTRUMENTATION_H
#define INSTRUMENTATION_H

#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>

#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <nvml.h>
#include <thread>
#include <string>


#ifdef TIME_INST
#define PRE_KERNEL __pre_kernel_time_inst();
#define POST_KERNEL __post_kernel_time_inst();
#define OUT_FILE_NAME "kerneltime.csv"
#endif

#ifdef POWER_INST
// This could be written by the students
#define PRE_KERNEL NVMLPowerMeter::get().start(); \
  for (auto start = std::chrono::steady_clock::now(), \
     nvml_now = start; nvml_now < start + std::chrono::milliseconds{1000}; \
     nvml_now = std::chrono::steady_clock::now()) {

// This could be written by the students
#define POST_KERNEL cudaDeviceSynchronize(); \
  } \
NVMLPowerMeter::get().stop();

#define OUT_FILE_NAME "kernelpower.csv"
#endif

// The started variable is used to prevent runnig the pre kernel routine more
// than once before calling the post kernel routine.
__attribute__((weak)) extern bool started = false;
// File handle for output file. Needs to be global, so the instrumentation will
// share only one handle and thus use the same file.
__attribute__((weak)) extern FILE *outFH = NULL;
// Global variable for start time. Use it to calculate the time difference in
// the post kernel routine
__attribute__((weak)) extern unsigned long start = 0;
// Global kernel counter
__attribute__((weak)) extern unsigned long counter = 0;

static inline void __pre_kernel_time_inst() {
  if (started) {
    printf("Error! Pre-kernel instrumentation routine already started.\n");
    exit(-1);
    return;
  }

  // Check file handle and open file if not existing.
  if (outFH == NULL) {
    outFH = fopen(OUT_FILE_NAME, "w");
    if (outFH == NULL) {
      printf("Error! Could not open out file %s..\n", OUT_FILE_NAME);
      exit(-1);
    }
  }

  started = true;

  // Complete this section here
  struct timeval mytime;
  gettimeofday(&mytime, NULL);
  start = (unsigned long)mytime.tv_sec * 1e6 + (unsigned long)mytime.tv_usec;
  // End of section
}

static inline void __post_kernel_time_inst() {
  if (not started) {
    printf("Error! Pre-kernel routine not started.\n");
    exit(-1);
    return;
  }

  unsigned long diff = 0;
  // Complete this section here, store elapsed time in diff
  cudaDeviceSynchronize();

  struct timeval mytime;
  gettimeofday(&mytime, NULL);
  unsigned long stop =
      (unsigned long)mytime.tv_sec * 1e6 + (unsigned long)mytime.tv_usec;

  diff = stop - start;
  // End of section

  printf("Duration: %lu\n", diff);
  // Write to out file
  fprintf(outFH, "%lu, %lu\n", counter++, diff);
  // Flush buffers to immediately store results to disk.
  fflush(outFH);
  started = false;
}

#ifdef __CUDACC__
__global__ void vecAdd(int *A,int *B,int *C,int N)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i<N) C[i] = A[i] + B[i]; 
}

void warmup() {
  const int gridSize = 1024 * 8;
  const int blockSize = 1024;
  const int N = gridSize * blockSize;
  const size_t vecSize = N * sizeof(int);
  int *A,*B,*C;
  cudaMalloc(&A,vecSize);
  cudaMalloc(&B,vecSize);
  cudaMalloc(&C,vecSize);
  cudaMemset(A, 0x00, vecSize);
  cudaMemset(B, 0x01, vecSize);
  vecAdd<<<gridSize,blockSize>>>(A,B,C,N);
  cudaThreadSynchronize();
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}

#endif


struct powerInfo {
  bool terminated = false;
  nvmlDevice_t deviceID;
  double power = 0.0;
};

void powerPolling(powerInfo& pi)
{
  pi.power = 0.0;
  nvmlReturn_t result;
  double duration = 0.0;
  unsigned int powerLevel = 0;

  std::chrono::high_resolution_clock::time_point t_begin, t_old, t_new;


  t_old = std::chrono::high_resolution_clock::now();
  t_begin = t_old;

  while (!pi.terminated) {
    // Get power in milli-watts
    result = nvmlDeviceGetPowerUsage(pi.deviceID, &powerLevel);
    if (result != 0) {
      printf("Error getting power usage!\n");
      break;
    }
    t_new = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t_new - t_old)
            .count();
    t_old = t_new;
    pi.power += (powerLevel/1000.0)* duration;
  }
  pi.power /=
      std::chrono::duration_cast<std::chrono::microseconds>(t_old - t_begin)
          .count();
}

class NVMLPowerMeter {
private:
  std::thread thread;
  pthread_t powerPollThread;
  powerInfo pi;
  std::ofstream logfile;
  unsigned long counter = 0;
  NVMLPowerMeter() { 
    logfile.open (OUT_FILE_NAME);
    warmup();
  }

 ~NVMLPowerMeter() {
   logfile.close();
 }

public:
  static NVMLPowerMeter& get() {
    static NVMLPowerMeter _instance;
    return _instance;
  }

  void start() {
  pi.power = -1.0;
  pi.terminated = false;
  nvmlReturn_t result;
  nvmlEnableState_t pmMode;
  double duration = 0.0;
  unsigned int powerLevel = 0;

  result = nvmlInit();
  if (result != 0) printf("Error initializing NVML!\n");
  result = nvmlDeviceGetHandleByIndex(0, &pi.deviceID);
  if (result != 0) printf("Error getting device handle!\n");
  result = nvmlDeviceGetPowerManagementMode(pi.deviceID, &pmMode);
  if (result != 0) printf("Error getting power management mode!\n");
    thread = std::thread(powerPolling, std::ref(pi));
  }

  void stop() {
    pi.terminated = true;
    thread.join();
    if(nvmlShutdown() != 0) std::cerr << "Error shutting down NVML!\n";
    if(pi.power < 0) std::cerr << "Error reading power!\n";
    std::cout << "Power: " << pi.power << " Watts\n";
    logfile << counter++ << ", " << pi.power << "\n";
    logfile.flush();
  }
};
static inline void __pre_kernel_power_inst() {};
static inline void __post_kernel_power_inst() {};

#endif
