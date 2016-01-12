#include "TW_utils.h"
#include <device_launch_parameters.h>

#include <TW_MemManager.h>
#include <helper_cuda.h>



namespace TW
{

	// !------------------------CUDA Kernel Functions-------------------------------
	
	/// \brief Compute the point (POI) position in each x, y direction
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
	__global__  void Precompute_POIPosition_kernel(
		// Input
		int_t iNumberX, int_t iNumberY,
		int_t iMarginX, int_t iMarginY,
		int_t iSubsetX, int_t iSubsetY,
		int_t iGridSpaceX, int_t iGridSpaceY,
		// Output
		int_t *d_iPXY)
	{
		auto tid = threadIdx.x + blockDim.x*blockIdx.x;
		auto i = tid / iNumberX;
		auto j = tid % iNumberX;

		if (tid<iNumberX*iNumberY)
		{
			d_iPXY[2 * tid] = iMarginY + iSubsetY + i * iGridSpaceY;
			d_iPXY[2 * tid + 1] = iMarginX + iSubsetX + j * iGridSpaceX;
		}
	}

	// ------------------------CUDA Kernel Functions End-----------------------------!

	// ------------------------CUDA Wrapper Functions--------------------------------
	void cuComputePOIPostions(
		// Output
		int_t *&Out_d_iPXY,						// Return the device handle
		// Inputs
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
		getLastCudaError("Error in calling Precompute_POIPosition_kernel");
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
		getLastCudaError("Error in calling Precompute_POIPosition_kernel");

		//!- Copy back the generated POI positions
		checkCudaErrors(cudaMemcpy(Out_h_iPXY, Out_d_iPXY, sizeof(int)*iNumberX*iNumberY * 2, cudaMemcpyDeviceToHost));
	}

	// ---------------------------CUDA Wrapper Functions End----------------------------!

}