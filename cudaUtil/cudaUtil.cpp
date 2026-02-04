#include "cudaUtil.h"


cuda::CudaDeviceInfo cuda::getDeviceInfo(int device)
{
	cuda::CudaDeviceInfo devInfo;

	cudaDeviceProp properties;
	cudaError_t err = cudaSuccess;

	err = cudaSetDevice(device);

	if(err) {
		throw cuda::CudaException(err);
	}

	err = cudaGetDeviceProperties(&properties, device);
	
	if(err) {
		throw cuda::CudaException(err);
	}

	devInfo.id = device;
	devInfo.major = properties.major;
	devInfo.minor = properties.minor;
	devInfo.mpCount = properties.multiProcessorCount;
	devInfo.mem = properties.totalGlobalMem;
	devInfo.name = std::string(properties.name);

	int cores = 0;
	switch(devInfo.major) {
	case 1:
		cores = 8;
		break;
	case 2:
        if(devInfo.minor == 0) {
            cores = 32;
        } else {
            cores = 48;
        }
		break;
	case 3:
		cores = 192;
		break;
	case 5:
		cores = 128;
		break;
	case 6:
        if(devInfo.minor == 1 || devInfo.minor == 2) {
            cores = 128;
        } else {
            cores = 64;
        }
        break;
	case 7:
		// Volta (sm_70) and Turing (sm_75)
		cores = 64;
		break;
	case 8:
		// Ampere (sm_80, sm_86, sm_87) and Ada Lovelace (sm_89)
		if(devInfo.minor == 0) {
			cores = 64;  // GA100 (A100)
		} else {
			cores = 128; // GA10x (A10, A40, etc.) and AD10x (L4, RTX 4090, etc.)
		}
		break;
	case 9:
		// Hopper (sm_90)
		cores = 128;
		break;
    default:
        cores = 128; // Assume modern architecture for future GPUs
        break;
	}
	devInfo.cores = cores;

	return devInfo;
}


std::vector<cuda::CudaDeviceInfo> cuda::getDevices()
{
	int count = getDeviceCount();

	std::vector<CudaDeviceInfo> devList;

	for(int device = 0; device < count; device++) {
		devList.push_back(getDeviceInfo(device));
	}

	return devList;
}

int cuda::getDeviceCount()
{
	int count = 0;

	cudaError_t err = cudaGetDeviceCount(&count);

    if(err) {
        throw cuda::CudaException(err);
    }

	return count;
}