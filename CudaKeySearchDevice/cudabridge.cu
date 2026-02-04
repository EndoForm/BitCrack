#include "cudabridge.h"

// function callKeyFinderKernel moved to CudaKeySearchDevice.cu

void waitForKernel() {
  // Check for kernel launch error
  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    throw cuda::CudaException(err);
  }

  // Wait for kernel to complete
  err = cudaDeviceSynchronize();
  fflush(stdout);
  if (err != cudaSuccess) {
    throw cuda::CudaException(err);
  }
}