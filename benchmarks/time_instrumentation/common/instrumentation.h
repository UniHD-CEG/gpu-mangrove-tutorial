#ifndef INSTRUMENTATION_H
#define INSTRUMENTATION_H

#define PRE_KERNEL __pre_kernel_time_inst();
#define POST_KERNEL __post_kernel_time_inst();

#define OUT_FILE_NAME "kerneltime.csv"

#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>

// The started variable is used to prevent runnig the pre kernel routine more than once before calling the post kernel routine.
__attribute__((weak)) extern bool started = false;
// File handle for output file. Needs to be global, so the instrumentation will share only one handle and thus use the same file.
__attribute__((weak)) extern FILE * outFH = NULL;
// Global variable for start time. Use it to calculate the time difference in the post kernel routine
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
      exit (-1);
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
  unsigned long stop = (unsigned long)mytime.tv_sec * 1e6 + (unsigned long)mytime.tv_usec;

  diff = stop - start;
  // End of section

  printf("Duration: %lu\n", diff);
  // Write to out file
  fprintf(outFH, "%lu, %lu\n", counter++, diff);
  // Flush buffers to immediately store results to disk.
  fflush(outFH);
  started = false;
}

#endif
