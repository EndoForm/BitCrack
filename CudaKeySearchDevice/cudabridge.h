#ifndef _BRIDGE_H
#define _BRIDGE_H

#include "cudaUtil.h"
#include "secp256k1.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

extern "C" {
void callKeyFinderKernel(int blocks, int threads, int points, bool useDouble,
                         int compression);
void callKeyFinderKernelFast(int blocks, int threads, int sharedMem,
                             int compression);
}

void waitForKernel();

cudaError_t setIncrementorPoint(const secp256k1::uint256 &x,
                                const secp256k1::uint256 &y);
cudaError_t allocateChainBuf(unsigned int count);
void cleanupChainBuf();

#endif