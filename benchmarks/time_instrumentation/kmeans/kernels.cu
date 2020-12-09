#include "instrumentation.h"

__global__ void kmeans_cluster_assignment(float *data, int *data_ca,
                                          float *centroids, int numData,
                                          int numCluster, int numDimensions) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= numData)
    return;

  float min_dist = INFINITY;
  int cluster_assignment = -1;
  for (int cluster = 0; cluster < numCluster; ++cluster) {
    float dist = 0;
    for (int dim = 0; dim < numDimensions; ++dim) {
      float square = data[tid * numDimensions + dim] -
                     centroids[cluster * numDimensions + dim];
      square *= square;
      dist += square;
    }
    dist = sqrt(dist);

    if (dist < min_dist) {
      min_dist = dist;
      cluster_assignment = cluster;
    }
  }
  data_ca[tid] = cluster_assignment;

  return;
}

void kmeans_cluster_assignment_wrapper(int grid_size, int block_size,
                                       float *data, int *data_ca,
                                       float *centroids, int numData,
                                       int numCluster, int numDimensions) {
  kmeans_cluster_assignment<<<grid_size, block_size>>>(
      data, data_ca, centroids, numData, numCluster, numDimensions);
}

__global__ void kmeans_centroid_sum(float *data, int *data_ca, float *centroids,
                                    int *cluster_count, int numData,
                                    int numCluster, int numDimensions) {
  extern __shared__ float shm[];
  float *s_data = (float *)shm;
  int *s_ca = (int *)(s_data + blockDim.x * numDimensions);

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numData)
    return;

  for (int dim = 0; dim < numDimensions; ++dim) {
    s_data[threadIdx.x * numDimensions + dim] = data[tid * numDimensions + dim];
  }
  s_ca[threadIdx.x] = data_ca[tid];

  __syncthreads();

  if (threadIdx.x < numCluster * numDimensions) {
    int cluster = threadIdx.x / numDimensions;
    int dim = threadIdx.x % numDimensions;
    float sum = 0.0;
    int count = 0;
    for (int i = 0; i < blockDim.x; ++i) {
      if (s_ca[i] == cluster) {
        ++count;
        sum += s_data[i * numDimensions + dim];
      }
    }
    atomicAdd(&centroids[cluster * numDimensions + dim], sum);
    if (dim == 0)
      atomicAdd(&cluster_count[cluster], count);
  }
}

void kmeans_centroid_sum_wrapper(int grid_size, int block_size, float *data,
                                 int *data_ca, float *centroids,
                                 int *cluster_count, int numData,
                                 int numCluster, int numDimensions) {
  kmeans_centroid_sum<<<grid_size, block_size,
                        block_size *(sizeof(int) +
                                     numDimensions * sizeof(float))>>>(
      data, data_ca, centroids, cluster_count, numData, numCluster,
      numDimensions);
}

__global__ void kmeans_centroid_update(float *centroids, int *cluster_count,
                                       int numCluster, int numDimensions) {
  if (threadIdx.x < numCluster * numDimensions) {
    int cluster = threadIdx.x / numDimensions;
    int dim = threadIdx.x % numDimensions;
    centroids[cluster * numDimensions + dim] /= cluster_count[cluster];
  }
}

void kmeans_centriod_update_wrapper(int grid_size, int block_size,
                                    float *centroids, int *cluster_count,
                                    int numCluster, int numDimensions) {
  kmeans_centroid_update<<<grid_size, block_size>>>(centroids, cluster_count,
                                                    numCluster, numDimensions);
}
