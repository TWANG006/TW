#ifndef TW_UTILS_H
#define	TW_UTILS_H

#include <cuda_runtime.h>
#include <cmath>

namespace TW
{
	//!- Return the 1D index of a 2D matrix
	inline __host__ __device__ int ELT2D(int x, int y, int width)
	{
		return (y*width + x);
	}

	//!- Return the 1D index of a 3D matrix
	inline __host__ __device__ int ELT3D(int x, int y, int z, int width, int height)
	{
		return ((z*height + y)*width + x);
	}

	// lerp
	inline __device__ __host__ float lerp(float a, float b, float t)
	{
		return a + t*(b - a);
	}

	// clamp
	inline __device__ __host__ float clamp(float f, float a, float b)
	{
		return fmaxf(a, fminf(f, b));
	}

} // namespace TW


#endif // !UTILS_H
