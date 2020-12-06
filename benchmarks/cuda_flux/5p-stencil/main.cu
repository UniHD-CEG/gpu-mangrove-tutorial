#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

// helper for time measurement
typedef std::chrono::duration<double, std::milli> d_ms;
const auto &now = std::chrono::high_resolution_clock::now;

// Define Error Checking Macro
#define CU_CHK(ERRORCODE)                                                      \
  {                                                                            \
    cudaError_t error = ERRORCODE;                                             \
    if (error != 0) {                                                          \
      std::cerr << cudaGetErrorName(error) << ": "                             \
                << cudaGetErrorString(error) << " at " << __FILE__ << ":"      \
                << __LINE__ << "\n";                                           \
    }                                                                          \
  }

// Constants
const static int DEFAULT_NUM_ELEMENTS = 1024;
const static int DEFAULT_NUM_ITERATIONS = 5;
const static int DEFAULT_BLOCK_DIM = 128;

// Structures
struct StencilArray_t {
  float *array;
  float *tmp_array;
  int size; // size == width == height
};

// Function Prototypes
void writeToFile(float *matrix, const char *name, int size);

// Stencil Code Kernel for the speed calculation
extern void simpleStencil_Kernel_Wrapper(int gridSize, int blockSize, int size,
                                         float *grid_old, float *grid_new);
extern void optStencil_Kernel_Wrapper(int gridSize, int blockSize, int shm_size,
                                      int size, float *grid_old,
                                      float *grid_new);
// Main
int main(int argc, char *argv[]) {
  // Process Arguments
  if (argc < 2 or std::string(argv[1]) == "-h") {
    std::cout << "Usage:\n\t" << argv[0] << " <problemsize> [iterations] [result.pgm]\n";
    return 1;
  }

  int numElements = 0;
  numElements = std::stoi(argv[1]);
  numElements = numElements > 0 ? numElements : DEFAULT_NUM_ELEMENTS;

  int numIterations = 0;
  if (argc > 2)
    numIterations = std::stoi(argv[2]);
  numIterations = numIterations != 0 ? numIterations : DEFAULT_NUM_ITERATIONS;

  // Allocate Memory

  // Host Memory
  StencilArray_t h_array;
  h_array.size = numElements;
  // Pinned Memory
  CU_CHK(cudaMallocHost(
      &(h_array.array),
      static_cast<size_t>(h_array.size * h_array.size * sizeof(float))));
  CU_CHK(cudaMallocHost(
      &(h_array.tmp_array),
      static_cast<size_t>(h_array.size * h_array.size * sizeof(float))));

  // Init Particles
  //  srand(static_cast<unsigned>(time(0)));
  srand(0); // Always the same random numbers
  for (int i = 0; i < h_array.size; i++) {
    for (int j = 0; j < h_array.size; j++) {
      if (i == 0)
        h_array.array[i + h_array.size * j] = 127;
      else if (i > h_array.size / 4 && i < h_array.size * 3 / 4 &&
               j > h_array.size / 4 && j < h_array.size * 3 / 4)
        h_array.array[i + h_array.size * j] = 100;
      else
        h_array.array[i + h_array.size * j] = 0;
    }
  }

  // Device Memory
  StencilArray_t d_array;
  d_array.size = h_array.size;
  CU_CHK(cudaMalloc(
      &(d_array.array),
      static_cast<size_t>(d_array.size * d_array.size * sizeof(float))));
  CU_CHK(cudaMalloc(
      &(d_array.tmp_array),
      static_cast<size_t>(d_array.size * d_array.size * sizeof(float))));

  // Copy Data to the Device
  auto t1 = now();
  CU_CHK(cudaMemcpy(
      d_array.array, h_array.array,
      static_cast<size_t>(d_array.size * d_array.size * sizeof(float)),
      cudaMemcpyHostToDevice));
  CU_CHK(cudaMemcpy(
      d_array.tmp_array, h_array.array,
      static_cast<size_t>(d_array.size * d_array.size * sizeof(float)),
      cudaMemcpyHostToDevice));
  auto t2 = now();

  // Block Dimension / Threads per Block
  int block_dim = DEFAULT_BLOCK_DIM;

#ifdef OPT_KERNEL
  std::cout << "Using optimized Kernel\n";
#endif
  int grid_dim = ceil(static_cast<float>(d_array.size) /
                      static_cast<float>(block_dim - 2));

  std::cout << "Computing grid with " << d_array.size << "x" << d_array.size
            << " elements and " << numIterations << " iterations\n";
  std::cout << "Launch kernel with " << grid_dim << " blocks and " << block_dim
            << " threads per block\n";

  auto t3 = now();
  for (int i = 0; i < numIterations; i++) {
#ifdef OPT_KERNEL
    optStencil_Kernel_Wrapper(grid_dim, block_dim,
                              block_dim * 4 * sizeof(float), d_array.size,
                              d_array.array, d_array.tmp_array);
#else
    simpleStencil_Kernel_Wrapper(grid_dim, block_dim, d_array.size,
                                 d_array.array, d_array.tmp_array);
#endif

    float *tmp = d_array.array;
    d_array.array = d_array.tmp_array;
    d_array.tmp_array = tmp;
  }

  // Synchronize
  cudaDeviceSynchronize();
  auto t4 = now();

  // Copy Back Data
  auto t5 = now();
  cudaMemcpy(h_array.array, d_array.array,
             static_cast<size_t>(h_array.size * h_array.size * sizeof(float)),
             cudaMemcpyDeviceToHost);
  auto t6 = now();

  if(argc > 3)
  writeToFile(h_array.array, argv[3], h_array.size);

  // Compute time for copies and kernel
  d_ms time_copyH2D = t2 - t1;
  d_ms time_kernel = t4 - t3;
  d_ms time_copyD2H = t6 - t5;

  // Free Memory
  cudaFreeHost(h_array.array);
  cudaFreeHost(h_array.tmp_array);

  cudaFree(d_array.array);
  cudaFree(d_array.tmp_array);

  // Print Meassurement Results
  std::cout << "Results:\n"
            << "H2D \[s], kernel [s], D2H [s]\n"
            << time_copyH2D.count() << ", " << time_kernel.count() << ", "
            << time_copyD2H.count() << "\n";
  return 0;
}

void writeToFile(float *grid, const char *name, int size) {
  FILE *pFile;

  pFile = fopen(name, "w");
  int i, j;

  fprintf(pFile, "P2 %d %d %d\n", size, size, 127);

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      fprintf(pFile, "%d ", (int)grid[j * size + i]);
    }
    fprintf(pFile, "\n");
  }

  fclose(pFile);

  return;
}
