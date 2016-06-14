#include "TW_cuTempUtils.h"

#include "TW.h"
#include <cuda_runtime.h>

namespace TW{
/// \brief CUDA kernel to initialize an array
///
/// \param devPtr device pointer holding the array
/// \param val the value used to initialize the array elements
/// \param nwords number of bytes to be initialized
template<typename T>
__global__ void initKernel(T * devPtr, const T val, const size_t nwords)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; tidx < nwords; tidx += stride)
        devPtr[tidx] = val;
}


template<typename T>
void cuInitialize(T* devPtr, const T val, const size_t nwords)
{
	initKernel<T><<<256, 64>>>(devPtr, val, nwords);
}

template TW_LIB_DLL_EXPORTS void cuInitialize<float>(float *devPtr, const float val, const size_t nwords);
template TW_LIB_DLL_EXPORTS void cuInitialize<double>(double *devPtr, const double val, const size_t nwords);
template TW_LIB_DLL_EXPORTS void cuInitialize<int>(int *devPtr, const int val, const size_t nwords);
template TW_LIB_DLL_EXPORTS void cuInitialize<unsigned int>(unsigned int *devPtr, const unsigned int val, const size_t nwords);
template TW_LIB_DLL_EXPORTS void cuInitialize<uchar1>(uchar1 *devPtr, const uchar1 val, const size_t nwords);
template TW_LIB_DLL_EXPORTS void cuInitialize<uchar2>(uchar2 *devPtr, const uchar2 val, const size_t nwords);
template TW_LIB_DLL_EXPORTS void cuInitialize<uchar3>(uchar3 *devPtr, const uchar3 val, const size_t nwords);
template TW_LIB_DLL_EXPORTS void cuInitialize<uchar4>(uchar4 *devPtr, const uchar4 val, const size_t nwords);


}