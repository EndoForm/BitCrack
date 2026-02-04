#include "CudaKeySearchDevice.h"
#include "KeySearchTypes.h"
#include "ptx.cuh"
#include "secp256k1.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ripemd160.cuh"
#include "sha256.cuh"

#include "secp256k1.h"

#include "CudaAtomicList.cuh"
#include "CudaDeviceKeys.cuh"
#include "CudaHashLookup.cuh"

__constant__ unsigned int _INC_X[8];

__constant__ unsigned int _INC_Y[8];

__constant__ unsigned int *_CHAIN[1];

static unsigned int *_chainBufferPtr = NULL;

__device__ void doRMD160FinalRound(const unsigned int hIn[5],
                                   unsigned int hOut[5]) {
  const unsigned int iv[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476,
                              0xc3d2e1f0};

  for (int i = 0; i < 5; i++) {
    hOut[i] = endian(hIn[i] + iv[(i + 1) % 5]);
  }
}

/**
 * Allocates device memory for storing the multiplication chain used in
 the batch inversion operation
 */
cudaError_t allocateChainBuf(unsigned int count) {
  cudaError_t err =
      cudaMalloc(&_chainBufferPtr, (size_t)count * sizeof(unsigned int) * 8);

  if (err) {
    return err;
  }

  err = cudaMemcpyToSymbol(_CHAIN, &_chainBufferPtr, sizeof(unsigned int *));
  if (err) {
    cudaFree(_chainBufferPtr);
  }

  return err;
}

void cleanupChainBuf() {
  if (_chainBufferPtr != NULL) {
    cudaFree(_chainBufferPtr);
    _chainBufferPtr = NULL;
  }
}

/**
 *Sets the EC point which all points will be incremented by
 */
cudaError_t setIncrementorPoint(const secp256k1::uint256 &x,
                                const secp256k1::uint256 &y) {
  unsigned int xWords[8];
  unsigned int yWords[8];

  x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
  y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

  cudaError_t err =
      cudaMemcpyToSymbol(_INC_X, xWords, sizeof(unsigned int) * 8);
  if (err) {
    return err;
  }

  return cudaMemcpyToSymbol(_INC_Y, yWords, sizeof(unsigned int) * 8);
}

__device__ void hashPublicKey(const unsigned int *x, const unsigned int *y,
                              unsigned int *digestOut) {
  unsigned int hash[8];

  sha256PublicKey(x, y, hash);

  // Swap to little-endian
  for (int i = 0; i < 8; i++) {
    hash[i] = endian(hash[i]);
  }

  ripemd160sha256NoFinal(hash, digestOut);
}

__device__ void hashPublicKeyCompressed(const unsigned int *x,
                                        unsigned int yParity,
                                        unsigned int *digestOut) {
  unsigned int hash[8];

  sha256PublicKeyCompressed(x, yParity, hash);

  // Swap to little-endian
  for (int i = 0; i < 8; i++) {
    hash[i] = endian(hash[i]);
  }

  ripemd160sha256NoFinal(hash, digestOut);
}

__device__ void setResultFound(int idx, bool compressed, unsigned int x[8],
                               unsigned int y[8], unsigned int digest[5],
                               int step) {
  CudaDeviceResult r;
  r.step = step;

  r.block = blockIdx.x;
  r.thread = threadIdx.x;
  r.idx = idx;
  r.compressed = compressed;

  for (int i = 0; i < 8; i++) {
    r.x[i] = x[i];
    r.y[i] = y[i];
  }

  doRMD160FinalRound(digest, r.digest);

  atomicListAdd(&r, sizeof(r));
}

__device__ void doIteration(int pointsPerThread, int compression) {
  unsigned int *chain = _CHAIN[0];
  unsigned int *xPtr = ec::getXPtr();
  unsigned int *yPtr = ec::getYPtr();

  // Multiply together all (_Gx - x) and then invert
  unsigned int inverse[8] = {0, 0, 0, 0, 0, 0, 0, 1};
  for (int i = 0; i < pointsPerThread; i++) {
    unsigned int x[8];

    unsigned int digest[5];

    readInt(xPtr, i, x);

    if (compression == PointCompressionType::UNCOMPRESSED ||
        compression == PointCompressionType::BOTH) {
      unsigned int y[8];
      readInt(yPtr, i, y);

      hashPublicKey(x, y, digest);

      if (checkHash(digest)) {
        setResultFound(i, false, x, y, digest, 0);
      }
    }

    if (compression == PointCompressionType::COMPRESSED ||
        compression == PointCompressionType::BOTH) {
      hashPublicKeyCompressed(x, readIntLSW(yPtr, i), digest);

      if (checkHash(digest)) {
        unsigned int y[8];
        readInt(yPtr, i, y);
        setResultFound(i, true, x, y, digest, 0);
      }
    }

    beginBatchAdd(_INC_X, x, chain, i, i, inverse);
  }

  doBatchInverse(inverse);

  for (int i = pointsPerThread - 1; i >= 0; i--) {

    unsigned int newX[8];
    unsigned int newY[8];

    completeBatchAdd(_INC_X, _INC_Y, xPtr, yPtr, i, i, chain, inverse, newX,
                     newY);

    writeInt(xPtr, i, newX);
    writeInt(yPtr, i, newY);
  }
}

__device__ void doIterationWithDouble(int pointsPerThread, int compression) {
  unsigned int *chain = _CHAIN[0];
  unsigned int *xPtr = ec::getXPtr();
  unsigned int *yPtr = ec::getYPtr();

  // Multiply together all (_Gx - x) and then invert
  unsigned int inverse[8] = {0, 0, 0, 0, 0, 0, 0, 1};
  for (int i = 0; i < pointsPerThread; i++) {
    unsigned int x[8];

    unsigned int digest[5];

    readInt(xPtr, i, x);

    // uncompressed
    if (compression == PointCompressionType::UNCOMPRESSED ||
        compression == PointCompressionType::BOTH) {
      unsigned int y[8];
      readInt(yPtr, i, y);
      hashPublicKey(x, y, digest);

      if (checkHash(digest)) {
        setResultFound(i, false, x, y, digest, 0);
      }
    }

    // compressed
    if (compression == PointCompressionType::COMPRESSED ||
        compression == PointCompressionType::BOTH) {

      hashPublicKeyCompressed(x, readIntLSW(yPtr, i), digest);

      if (checkHash(digest)) {

        unsigned int y[8];
        readInt(yPtr, i, y);

        setResultFound(i, true, x, y, digest, 0);
      }
    }

    beginBatchAddWithDouble(_INC_X, _INC_Y, xPtr, chain, i, i, inverse);
  }

  doBatchInverse(inverse);

  for (int i = pointsPerThread - 1; i >= 0; i--) {

    unsigned int newX[8];
    unsigned int newY[8];

    completeBatchAddWithDouble(_INC_X, _INC_Y, xPtr, yPtr, i, i, chain, inverse,
                               newX, newY);

    writeInt(xPtr, i, newX);
    writeInt(yPtr, i, newY);
  }
}

// Fast kernel constants
#define FAST_POINTS 4
#define FAST_STEPS 1024

__device__ void doIterationFast(int compression) {
  extern __shared__ unsigned int s_chain[];

  unsigned int *xPtr = ec::getXPtr();
  unsigned int *yPtr = ec::getYPtr();

  // Registers for points
  unsigned int rx[FAST_POINTS][8];
  unsigned int ry[FAST_POINTS][8];

  // Load from global to registers
  for (int i = 0; i < FAST_POINTS; i++) {
    readInt(xPtr, i, rx[i]);
    readInt(yPtr, i, ry[i]);
  }

  // Iterate without global memory access
  for (int step = 0; step < FAST_STEPS; step++) {

    unsigned int inverse[8] = {0, 0, 0, 0, 0, 0, 0, 1};

    // 1. Accumulate product
    for (int i = 0; i < FAST_POINTS; i++) {
      // Check hash (for the current point)
      unsigned int digest[5];
      if (compression == PointCompressionType::UNCOMPRESSED ||
          compression == PointCompressionType::BOTH) {
        hashPublicKey(rx[i], ry[i], digest);
        if (checkHash(digest))
          setResultFound(i, false, rx[i], ry[i], digest, step);
      }
      if (compression == PointCompressionType::COMPRESSED ||
          compression == PointCompressionType::BOTH) {
        hashPublicKeyCompressed(rx[i], ry[i][7] & 1, digest);
        if (checkHash(digest))
          setResultFound(i, true, rx[i], ry[i], digest, step);
      }

      // Begin batch add (using registers and shared chain)
      beginBatchAddWithDoubleSharedReg(_INC_X, _INC_Y, rx[i], s_chain, i,
                                       inverse);
    }

    // 2. Invert
    doBatchInverse(inverse);

    // 3. Backtrack / Complete
    for (int i = FAST_POINTS - 1; i >= 0; i--) {
      unsigned int newX[8];
      unsigned int newY[8];

      completeBatchAddSharedReg(_INC_X, _INC_Y, rx[i], ry[i], i, s_chain,
                                inverse, newX, newY);

      // Update registers
      copyBigInt(newX, rx[i]);
      copyBigInt(newY, ry[i]);
    }
  }

  // Write back to global
  for (int i = 0; i < FAST_POINTS; i++) {
    writeInt(xPtr, i, rx[i]);
    writeInt(yPtr, i, ry[i]);
  }
}

/**
 * Performs a single iteration
 */
__global__ void keyFinderKernel(int points, int compression) {
  doIteration(points, compression);
}

__global__ void keyFinderKernelWithDouble(int points, int compression) {
  doIterationWithDouble(points, compression);
}

__global__ void keyFinderKernelFast(int compression) {
  doIterationFast(compression);
}

extern "C" {
void callKeyFinderKernel(int blocks, int threads, int pointsPerThread,
                         bool useDouble, int compression) {
  if (useDouble) {
    keyFinderKernelWithDouble<<<blocks, threads>>>(pointsPerThread,
                                                   compression);
  } else {
    keyFinderKernel<<<blocks, threads>>>(pointsPerThread, compression);
  }
  cudaDeviceSynchronize();
}

void callKeyFinderKernelFast(int blocks, int threads, int sharedMem,
                             int compression) {
  keyFinderKernelFast<<<blocks, threads, sharedMem>>>(compression);
  cudaDeviceSynchronize();
}
}