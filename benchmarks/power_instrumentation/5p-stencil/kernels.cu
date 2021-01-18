#include "instrumentation.h"

// Stencil Code Kernel for the heat calculation
__global__ void simpleStencil_Kernel(int size, float *grid_old,
                                     float *grid_new) {
  const float gamma = 0.24;
  float tmp_val;

  for (int Id = blockIdx.x * blockDim.x + threadIdx.x;
       Id < (size - 2) * (size - 2); Id += blockDim.x + gridDim.x) {
    int index = Id + 1 + size +
                ((Id) / (size - 2)) * 2; // + 1 + size um vom rand wegzukommen

    tmp_val = grid_old[index] +
              gamma * (-4 * grid_old[index] + grid_old[index + 1] +
                       grid_old[index - 1] + grid_old[index + size] +
                       grid_old[index - size]);

    if (tmp_val > 127)
      tmp_val = 127.0;
    if (tmp_val < 0)
      tmp_val = 0.0;

    grid_new[index] = tmp_val;
  }
}

void simpleStencil_Kernel_Wrapper(int gridSize, int blockSize, int size,
                                  float *grid_old, float *grid_new) {
  dim3 grid_dim = dim3(gridSize);
  dim3 block_dim = dim3(blockSize);

  
  simpleStencil_Kernel<<<grid_dim, block_dim>>>(size, grid_old, grid_new);
  
}

// optimized Stencil Code Kernel for the heat calculation
__global__ void optStencil_Kernel(int size, float *grid_old, float *grid_new) {
  const float gamma = 0.24;
  extern __shared__ float shm[];
  float tmp_val;
  float curr_old[4];
  int rot_index_e;
  int rot_index_c;
  int rot_index_w;

  int Idy = blockIdx.x * blockDim.x + threadIdx.x - blockIdx.x * 2;

  if (Idy < size - 1) {
    for (int x = 0; x < size; x++) {
      rot_index_e = x % 4;
      rot_index_c = (x + 3) % 4;
      rot_index_w = (x + 2) % 4;

      curr_old[rot_index_e] = grid_old[x + Idy * size];
      shm[(rot_index_e)*blockDim.x + threadIdx.x] = curr_old[rot_index_e];

      __syncthreads();

      if (x > 1 && threadIdx.x != 0 && threadIdx.x != (blockDim.x - 1)) {
        tmp_val =
            curr_old[rot_index_c] +
            gamma * (-4 * curr_old[rot_index_c] +
                     shm[rot_index_e * blockDim.x + threadIdx.x] +     // east
                     shm[rot_index_w * blockDim.x + threadIdx.x] +     // west
                     shm[rot_index_c * blockDim.x + threadIdx.x - 1] + // north
                     shm[rot_index_c * blockDim.x + threadIdx.x + 1]   // south
                    );

        if (tmp_val > 127)
          tmp_val = 127.0;
        if (tmp_val < 0)
          tmp_val = 0.0;

        grid_new[(x - 1) + Idy * size] = tmp_val;
      }

      __syncthreads();
    }
  }
}

void optStencil_Kernel_Wrapper(int gridSize, int blockSize, int shm_size,
                               int size, float *grid_old, float *grid_new) {
  dim3 grid_dim = dim3(gridSize);
  dim3 block_dim = dim3(blockSize);

  
  optStencil_Kernel<<<grid_dim, block_dim, shm_size>>>(size, grid_old,
                                                       grid_new);
  
}
