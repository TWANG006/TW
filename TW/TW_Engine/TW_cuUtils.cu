#include "TW_utils.h"
#include <device_launch_parameters.h>

#include <TW_MemManager.h>
#include <helper_cuda.h>



namespace TW
{

// !------------------------CUDA Kernel Functions-------------------------------

/// \brief compute z = ax + y in parallel.
/// \ strided for loop is used for the optimized performance and kernel size 
/// \ flexibility. NOTE: results are saved in vector y

/// \param n number of elements of x and y vector
/// \param a multiplier
/// \param x, y input vectors
__global__ void saxpy(// Inputs
					  int n, 
					  real_t a, 
					  real_t *x,
					  // Output
					  real_t *y)
{
	for(auto i = blockIdx.x*blockDim.x+threadIdx.x;
		     i < n;
			 i += blockDim.x * gridDim.x)
	{
		y[i] = a * x[i] + y[i];
	}
}

/// \brief CUDA kernel to Compute the point (POI) position in each x, y direction within
/// the ROI area
///
/// \param iNumberX number of POIs in x direction
/// \param iNumberY number of POIs in y direction
/// \param iMarginX number of extra safe pixels at ROI boundary in x direction
/// \param iMarginY number of extra safe pixels at ROI boundary in y direction
/// \param iSubsetX half size of the square subset in x direction
/// \param iSubsetY half size of the square subset in y direction
/// \param iGridSpaceX number of pixels between two POIs in x direction
/// \param iGirdSpaceY number of pixels between two POIs in y direction
/// \param d_iPXY positions of iPXY to be computed
__global__  void Precompute_POIPosition_kernel(// Input
											   int_t iNumberX,    int_t iNumberY,
											   int_t iMarginX,    int_t iMarginY,
											   int_t iSubsetX,    int_t iSubsetY,
											   int_t iGridSpaceX, int_t iGridSpaceY,
											   // Output
											   int_t *d_iPXY)
{
	auto tid = threadIdx.x + blockDim.x*blockIdx.x;
	auto i = tid / iNumberX;
	auto j = tid % iNumberX;

	if (tid < iNumberX*iNumberY)
	{
		d_iPXY[2 * tid] = iMarginY + iSubsetY + i * iGridSpaceY;
		d_iPXY[2 * tid + 1] = iMarginX + iSubsetX + j * iGridSpaceX;
	}
}

/// \brief CUDA kernel to Compute the point (POI) position in each x, y direction within
/// the whole image
///
/// \param iNumberX number of POIs in x direction
/// \param iNumberY number of POIs in y direction
/// \param iMarginX number of extra safe pixels at ROI boundary in x direction
/// \param iMarginY number of extra safe pixels at ROI boundary in y direction
/// \param iSubsetX half size of the square subset in x direction
/// \param iSubsetY half size of the square subset in y direction
/// \param iGridSpaceX number of pixels between two POIs in x direction
/// \param iGirdSpaceY number of pixels between two POIs in y direction
/// \param d_iPXY positions of iPXY to be computed
__global__  void Precompute_POIPosition_WholeImg_kernel(// Input
														int_t iStartX,     int_t iStartY,
														int_t iNumberX,    int_t iNumberY,
														int_t iMarginX,    int_t iMarginY,
														int_t iSubsetX,    int_t iSubsetY,
														int_t iGridSpaceX, int_t iGridSpaceY,
														// Output
														int_t *d_iPXY)
{
	auto tid = threadIdx.x + blockDim.x*blockIdx.x;
	auto i = tid / iNumberX;
	auto j = tid % iNumberX;

	if (tid < iNumberX*iNumberY)
	{
		d_iPXY[2 * tid] = iStartX + iMarginY + iSubsetY + i * iGridSpaceY;
		d_iPXY[2 * tid + 1] = iStartY + iMarginX + iSubsetX + j * iGridSpaceX;
	}
}


// Parallel reduction utilities
//template <unsigned int blockSize>
//__device__ void sumWarpReduce(volatile int *sdata, unsigned int tid)
//{
//	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//	if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; 
//	if (blockSize >= 16) sdata[tid] += sdata[tid + 8]; 
//	if (blockSize >= 8) sdata[tid] += sdata[tid + 4]; 
//	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
//}
//template <unsigned int blockSize>
//__device__ void maxWarpReduce(volatile int *sdata, unsigned int tid)
//{
//	if (blockSize >= 64){ if(sdata[tid] < sdata[tid + 32]) sdata[tid] = sdata[tid+32];};
//	if (blockSize >= 32) sdata[tid] += sdata[tid + 16]; 
//	if (blockSize >= 16) sdata[tid] += sdata[tid + 8]; 
//	if (blockSize >= 8) sdata[tid] += sdata[tid + 4]; 
//	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
//}
//
//template <unsigned int blockSize> 
//__global__ void sumReduce(int *g_idata, int *g_odata, unsigned int n) 
//{
//	extern __shared__ int sdata[]; 
//	unsigned int tid = threadIdx.x; 
//	unsigned int i = blockIdx.x*(blockSize * 2) + tid; 
//	unsigned int gridSize = blockSize * 2 * gridDim.x; 
//	sdata[tid] = 0;
//	while (i < n) 
//	{
//		sdata[tid] += g_idata[i] + g_idata[i + blockSize]; 
//		i += gridSize; 
//	}
//	__syncthreads();
//	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
//	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); } 
//	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
//	if (tid < 32) warpReduce(sdata, tid); if (tid == 0) g_odata[blockIdx.x] = sdata[0];
//}
//
// ------------------------CUDA Kernel Functions End-----------------------------!


// ------------------------CUDA Wrapper Functions--------------------------------

void cuComputePOIPositions(// Output
						   int_t *&Out_d_iPXY,			// Return the device handle
						   // Inputs
						   int_t iNumberX, int_t iNumberY,
						   int_t iMarginX, int_t iMarginY,
						   int_t iSubsetX, int_t iSubsetY,
						   int_t iGridSpaceX, int_t iGridSpaceY)
{
	//!- Allocate Memory for device
	checkCudaErrors(cudaMalloc((void**)&Out_d_iPXY, 
							   sizeof(int)*iNumberX*iNumberY * 2));

	dim3 gridDim((iNumberX*iNumberY + BLOCK_SIZE_256 - 1) / (BLOCK_SIZE_256)); 

	//!- Launch the kernel
	Precompute_POIPosition_kernel<<<gridDim, BLOCK_SIZE_256 >>>(iNumberX,	iNumberY,
																iMarginX,	iMarginY,
																iSubsetX,	iSubsetY,
																iGridSpaceX,iGridSpaceY,
																Out_d_iPXY);
	getLastCudaError("Error in calling Precompute_POIPosition_kernel");
}

void cuComputePOIPositions(// Outputs
						   int_t *&Out_d_iPXY,						// Return the device handle
						   int_t *&Out_h_iPXY,						// Retrun the host handle
						   // Inputs
						   int_t iNumberX, int_t iNumberY,
						   int_t iMarginX, int_t iMarginY,
						   int_t iSubsetX, int_t iSubsetY,
						   int_t iGridSpaceX, int_t iGridSpaceY)
{
	//!- Allocate Memory for host & device
	hcreateptr<int_t>(Out_h_iPXY, sizeof(int)*iNumberX*iNumberY * 2);
	checkCudaErrors(cudaMalloc((void**)&Out_d_iPXY, sizeof(int)*iNumberX*iNumberY * 2));

	dim3 gridDim((iNumberX*iNumberY + BLOCK_SIZE_256 - 1) / (BLOCK_SIZE_256)); 

	//!- Launch the kernel
	Precompute_POIPosition_kernel<<<gridDim, BLOCK_SIZE_256 >>>(iNumberX,	iNumberY,
																iMarginX,	iMarginY,
																iSubsetX,	iSubsetY,
																iGridSpaceX,iGridSpaceY,
																Out_d_iPXY);
	getLastCudaError("Error in calling Precompute_POIPosition_kernel");

	//!- Copy back the generated POI positions
	checkCudaErrors(cudaMemcpy(Out_h_iPXY, 
							   Out_d_iPXY, 
							   sizeof(int)*iNumberX*iNumberY * 2, 
							   cudaMemcpyDeviceToHost));
}

void cuComputePOIPositions(// Outputs
						   int_t *&Out_d_iPXY,			// Return the device handle
						   int_t *&Out_h_iPXY,			// Retrun the host handle
						   // Inputs
						   int_t iStartX, int_t iStartY, // Start top-left point of the ROI
						   int_t iNumberX, int_t iNumberY,
						   int_t iMarginX, int_t iMarginY,
						   int_t iSubsetX, int_t iSubsetY,
						   int_t iGridSpaceX, int_t iGridSpaceY)
{
	//!- Allocate Memory for host & device
	hcreateptr<int_t>(Out_h_iPXY, sizeof(int)*iNumberX*iNumberY * 2);
	checkCudaErrors(cudaMalloc((void**)&Out_d_iPXY, 
							   sizeof(int)*iNumberX*iNumberY * 2));

	dim3 gridDim((iNumberX*iNumberY + BLOCK_SIZE_256 - 1) / (BLOCK_SIZE_256)); 

	//!- Launch the kernel
	Precompute_POIPosition_WholeImg_kernel<<<gridDim, BLOCK_SIZE_256 >>>(iStartX,     iStartY,
																	     iNumberX,	  iNumberY,
																		 iMarginX,	  iMarginY,
																		 iSubsetX,	  iSubsetY,
																		 iGridSpaceX, iGridSpaceY,
																		 Out_d_iPXY);
	getLastCudaError("Error in calling Precompute_POIPosition_WholeImg_kernel");

	//!- Copy back the generated POI positions
	checkCudaErrors(cudaMemcpy(Out_h_iPXY, 
							   Out_d_iPXY, 
							   sizeof(int)*iNumberX*iNumberY * 2, 
							   cudaMemcpyDeviceToHost));
}

void cuComputePOIPositions(// Outputs
						   int_t *&Out_d_iPXY,			// Retrun the device handle
						   // Inputs
						   int_t iStartX, int_t iStartY, // Start top-left point of the ROI
						   int_t iNumberX, int_t iNumberY,
						   int_t iMarginX, int_t iMarginY,
						   int_t iSubsetX, int_t iSubsetY,
						   int_t iGridSpaceX, int_t iGridSpaceY)
{
	//!- Allocate Memory for device
	checkCudaErrors(cudaMalloc((void**)&Out_d_iPXY, 
							   sizeof(int)*iNumberX*iNumberY * 2));

	dim3 gridDim((iNumberX*iNumberY + BLOCK_SIZE_256 - 1) / (BLOCK_SIZE_256)); 

	//!- Launch the kernel
	Precompute_POIPosition_WholeImg_kernel<<<gridDim, BLOCK_SIZE_256 >>>(iStartX,     iStartY,
																	     iNumberX,	  iNumberY,
																		 iMarginX,	  iMarginY,
																		 iSubsetX,	  iSubsetY,
																		 iGridSpaceX, iGridSpaceY,
																		 Out_d_iPXY);
	getLastCudaError("Error in calling Precompute_POIPosition_WholeImg_kernel");
}

void cuSaxpy(// Inputs
			 int_t n, 
			 real_t a, 
			 real_t *x,
			 int_t devID,
			 // Output
			 real_t *y)
{
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount,devID);

	printf("%d", numSMs);

	saxpy<<<32*numSMs, 256>>>(n,
							  a,
							  x,
							  y);
}


// ---------------------------CUDA Wrapper Functions End----------------------------!

} //!- namespace TW