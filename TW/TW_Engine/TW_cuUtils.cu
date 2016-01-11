#include "TW_utils.h"
#include <device_launch_parameters.h>

#include <TW_MemManager.h>
#include <helper_cuda.h>

//!- Compute the point (POI) position in each x, y direction
__global__  void Precompute_POIPosition_kernel(
	int iNumberX, int iNumberY,
	int iMarginX, int iMarginY,
	int iSubsetX, int iSubsetY,
	int iGridSpaceX, int iGridSpaceY,
	int *d_iPXY)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	int i = tid / iNumberX;
	int j = tid % iNumberX;

	if (tid<iNumberX*iNumberY)
	{
		d_iPXY[2 * tid] = iMarginY + iSubsetY + i * iGridSpaceY;
		d_iPXY[2 * tid + 1] = iMarginX + iSubsetX + j * iGridSpaceX;
	}
}

namespace TW
{
	void cuComputePOIPositions(
		int_t *&Out_d_iPXY,
		int_t iNumberX, int_t iNumberY,
		int_t iMarginX, int_t iMarginY,
		int_t iSubsetX, int_t iSubsetY,
		int_t iGridSpaceX, int_t iGridSpaceY)
	{
		//!- Allocate Memory for device
		checkCudaErrors(cudaMalloc((void**)&Out_d_iPXY, sizeof(int)*iNumberX*iNumberY * 2));

		//!- Launch the kernel
		Precompute_POIPosition_kernel << <(iNumberX*iNumberY + BLOCK_SIZE_256 - 1) / (BLOCK_SIZE_256), BLOCK_SIZE_256 >> >(
			iNumberX, iNumberY,
			iMarginX, iMarginY,
			iSubsetX, iSubsetY,
			iGridSpaceX, iGridSpaceY,
			Out_d_iPXY);
	}

	void cuComputePOIPostions(
		// Outputs
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

		//!- Launch the kernel
		Precompute_POIPosition_kernel << <(iNumberX*iNumberY + BLOCK_SIZE_256 - 1) / (BLOCK_SIZE_256), BLOCK_SIZE_256 >> >(
			iNumberX, iNumberY,
			iMarginX, iMarginY,
			iSubsetX, iSubsetY,
			iGridSpaceX, iGridSpaceY,
			Out_d_iPXY);

		//!- Copy back the generated POI positions
		checkCudaErrors(cudaMemcpy(Out_h_iPXY, Out_d_iPXY, sizeof(int)*iNumberX*iNumberY * 2, cudaMemcpyDeviceToHost));
	}
}