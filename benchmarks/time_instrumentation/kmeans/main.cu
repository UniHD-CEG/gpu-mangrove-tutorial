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
const static int DEFAULT_NUM_DATA = 1024;
const static int DEFAULT_NUM_CLUSTER = 3;
const static int DEFAULT_DIMENSIONS = 2;
const static int DEFAULT_NUM_ITERATIONS = 5;
const static int DEFAULT_BLOCK_DIM = 1024;

extern void kmeans_cluster_assignment_wrapper(int grid_size, int block_size,
                                              float *data, int *data_ca,
                                              float *centroids, int numData,
                                              int numCluster,
                                              int numDimensions);
extern void kmeans_centroid_sum_wrapper(int grid_size, int block_size,
                                        float *data, int *data_ca,
                                        float *centroids, int *cluster_count,
                                        int numData, int numCluster,
                                        int numDimensions);
extern void kmeans_centriod_update_wrapper(int grid_size, int block_size,
                                           float *centroids, int *cluster_count,
                                           int numCluster, int numDimensions);

// Main
int main(int argc, char *argv[]) {
  // Process Arguments
  if (argc < 2 or std::string(argv[1]) == "-h") {
    std::cout << "Usage:\n\t" << argv[0]
              << " <data points> [iterations] [dimensions]\n";
    return 1;
  }

  int numData = 0;
  numData = std::stoi(argv[1]);
  numData = numData > 0 ? numData : DEFAULT_NUM_DATA;

  int numIterations = 0;
  if (argc > 2)
    numIterations = std::stoi(argv[2]);
  numIterations = numIterations != 0 ? numIterations : DEFAULT_NUM_ITERATIONS;

  int numDimensions = 0;
  if (argc > 3)
    numDimensions = std::stoi(argv[2]);
  numDimensions = numDimensions != 0 ? numDimensions : DEFAULT_DIMENSIONS;

  int numCluster = DEFAULT_NUM_CLUSTER;

  // Allocate Memory
  // Host
  float *h_data;
  CU_CHK(cudaMallocHost(&h_data,
                        (size_t)(numData * numDimensions * sizeof(float))));
  float *h_centroids;
  CU_CHK(cudaMallocHost(&h_centroids,
                        (size_t)(numCluster * numDimensions * sizeof(float))));

  // Init
  srand(0); // Always the same random numbers
  for (int i = 0; i < numData; ++i)
    for (int d = 0; d < numDimensions; ++d)
      h_centroids[i * numDimensions + d] = (float)rand() / (double)RAND_MAX;
  for (int c = 0; c < numCluster; ++c)
    for (int d = 0; d < numDimensions; ++d)
      h_centroids[c * numDimensions + d] = (float)rand() / (double)RAND_MAX;

  // Device Memory
  float *d_data;
  CU_CHK(cudaMalloc(&d_data,
                        (size_t)(numData * numDimensions * sizeof(float))));
  int *d_data_ca;
  CU_CHK(cudaMalloc(&d_data_ca, (size_t)(numData * sizeof(int))));
  float *d_centroids;
  CU_CHK(cudaMalloc(&d_centroids,
                        (size_t)(numCluster * numDimensions * sizeof(float))));
  int *d_cluster_count;
  CU_CHK(cudaMalloc(&d_cluster_count, (size_t)(numCluster * sizeof(int))));

  // Copy Data to the Device
  auto t1 = now();
  CU_CHK(cudaMemcpy(d_data, h_data,
                    (size_t)(numData * numDimensions * sizeof(float)),
                    cudaMemcpyHostToDevice));
  CU_CHK(cudaMemcpy(d_centroids, d_centroids,
                    (size_t)(numCluster * numDimensions * sizeof(float)),
                    cudaMemcpyHostToDevice));
  CU_CHK(cudaMemset(d_cluster_count, 0, numCluster * sizeof(int)));
  auto t2 = now();

  // Block Dimension / Threads per Block
  int block_dim = DEFAULT_BLOCK_DIM;

  int grid_dim =
      ceil(static_cast<float>(numData) / static_cast<float>(block_dim));

  std::cout << "Computing  kmeans with " << numData << " elements and "
            << numIterations << " iterations\n";
  std::cout << "Launch kernel with " << grid_dim << " blocks and " << block_dim
            << " threads per block\n";

  auto t3 = now();
  for (int i = 0; i < numIterations; i++) {
    kmeans_cluster_assignment_wrapper(grid_dim, block_dim, d_data, d_data_ca,
                                      d_centroids, numData, numCluster,
                                      numDimensions);
    CU_CHK(
        cudaMemset(d_centroids, 0, numCluster * numDimensions * sizeof(float)));
    CU_CHK(cudaMemset(d_cluster_count, 0, numCluster * sizeof(int)));
    kmeans_centroid_sum_wrapper(grid_dim, block_dim, d_data, d_data_ca,
                                d_centroids, d_cluster_count, numData,
                                numCluster, numDimensions);
    kmeans_centriod_update_wrapper(grid_dim, numCluster * numDimensions,
                                   d_centroids, d_cluster_count, numCluster,
                                   numDimensions);
  }

  // Synchronize
  CU_CHK(cudaDeviceSynchronize());
  auto t4 = now();

  // Compute time for copies and kernel
  d_ms time_copyH2D = t2 - t1;
  d_ms time_kernel = t4 - t3;

  // Free Memory
  CU_CHK(cudaFreeHost(h_data));
  CU_CHK(cudaFreeHost(h_centroids));

  CU_CHK(cudaFree(d_data));
  CU_CHK(cudaFree(d_centroids));
  CU_CHK(cudaFree(d_data_ca));
  CU_CHK(cudaFree(d_cluster_count));

  // Print Meassurement Results
  std::cout << "Results:\n"
            << "H2D [ms], kernel [ms]\n"
            << time_copyH2D.count() << ", " << time_kernel.count() << "\n";
  return 0;
}
