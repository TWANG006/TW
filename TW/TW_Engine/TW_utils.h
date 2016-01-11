#ifndef TW_UTILS_H
#define	TW_UTILS_H

#include "TW.h"

#include <cuda_runtime.h>
#include <cmath>

namespace TW
{
	/// \brief Return the 1D index of a 2D matrix
	/// \param x position in x direction in a 2D matrix
	/// \param y position in y direction in a 2D matrix
	/// \param width width (number of cols) of a 2D matrix
	inline __host__ __device__ int_t ELT2D(int_t x, int_t y, int_t width)
	{
		return (y*width + x);
	}

	/// \brief Return the 1D index of a 3D matrix
	/// \param x position in x direction in a 3D matrix
	/// \param y position in y direction in a 3D matrix
	/// \param z position in z direction in a 3D matrix
	/// \param width width of a 3D matrix
	/// \param height height of a 3D matrix
	inline __host__ __device__ int_t ELT3D(int_t x, int_t y, int_t z, int_t width, int_t height)
	{
		return ((z*height + y)*width + x);
	}

	/// \brief lerp function for 
	/// \param a lower bound 
	/// \param b upper bound
	/// \param lerp parameter
	inline __device__ __host__ real_t lerp(real_t a, real_t b, real_t t)
	{
		return a + t*(b - a);
	}

	/// \brief clamp function
	/// \param a lower bound
	/// \param b upper bound
	/// \param f value to be clamped
	inline __device__ __host__ real_t clamp(real_t f, real_t a, real_t b)
	{
#ifdef TW_USE_DOUBLE
		return fmax(a, fmin(f, b));
#else
		return fmaxf(a, fminf(f, b));
#endif // TW_USE_DOUBL
	}

	/// \brief Function to compute positions of POIs on GPU, the result is stored in 
	/// an device array. NOTE: No need to pre-allocate memory for the device pointer
	/// \param Out_d_iPXY position array in device memory
	/// \param iNumberX number of POIs in x direction
	/// \param iNumberY number of POIs in y direction
	/// \param iMarginX number of extra safe pixels at ROI boundary in x direction
	/// \param iMarginY number of extra safe pixels at ROI boundary in y direction
	/// \param iSubsetX half size of the square subset in x direction
	/// \param iSubsetY half size of the square subset in y direction
	/// \param iGridSpaceX number of pixels between two POIs in x direction
	/// \param iGirdSpaceY number of pixels between two POIs in y direction
	TW_LIB_DLL_EXPORTS void cuComputePOIPostions(
		// Output
		int_t *&Out_d_iPXY,						// Return the device handle
		// Inputs
		int_t iNumberX, int_t iNumberY,
		int_t iMarginX, int_t iMarginY,
		int_t iSubsetX, int_t iSubsetY,
		int_t iGridSpaceX, int_t iGridSpaceY);

	/// \brief Function to compute positions of POIs on GPU, the result is stored both in 
	/// an device array and a host array. NOTE: No need to pre-allocate memory for the two pointers.
	/// \param Out_d_iPXY position array in device memory
	/// \param Out_h_iPXY position array in host memory
	/// \param iNumberX number of POIs in x direction
	/// \param iNumberY number of POIs in y direction
	/// \param iMarginX number of extra safe pixels at ROI boundary in x direction
	/// \param iMarginY number of extra safe pixels at ROI boundary in y direction
	/// \param iSubsetX half size of the square subset in x direction
	/// \param iSubsetY half size of the square subset in y direction
	/// \param iGridSpaceX number of pixels between two POIs in x direction
	/// \param iGirdSpaceY number of pixels between two POIs in y direction
	TW_LIB_DLL_EXPORTS void cuComputePOIPostions(
		// Outputs
		int_t *&Out_d_iPXY,						// Return the device handle
		int_t *&Out_h_iPXY,						// Retrun the host handle
		// Inputs
		int_t iNumberX, int_t iNumberY,
		int_t iMarginX, int_t iMarginY,
		int_t iSubsetX, int_t iSubsetY,
		int_t iGridSpaceX, int_t iGridSpaceY);

} // namespace TW


#endif // !UTILS_H
