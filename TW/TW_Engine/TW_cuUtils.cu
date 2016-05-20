#include "TW_utils.h"
#include <device_launch_parameters.h>

#include <TW_MemManager.h>
#include <helper_cuda.h>



namespace TW
{

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__constant__ real_t c_dBicubicMatrix[16][16]={ 
		{	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0 }, 
		{	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0 }, 
		{  -3,	3,	0,	0, -2, -1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0 }, 
		{	2, -2,	0,	0,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0 }, 
		{	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0 }, 
		{	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0 },
		{	0,	0,	0,	0,	0,	0,	0,	0, -3,	3,	0,	0, -2, -1,	0,	0 }, 
		{	0,	0,	0,	0,	0,	0,	0,	0,	2, -2,	0,	0,	1,	1,	0,	0 }, 
		{  -3,	0,	3,	0,	0,	0,	0,	0, -2,	0, -1,	0,	0,	0,	0,	0 }, 
		{	0,	0,	0,	0, -3,	0,	3,	0,	0,	0,	0,	0, -2,  0, -1,	0 },
		{	9, -9, -9,	9,	6,	3, -6, -3,	6, -6,	3, -3,	4,	2,	2,	1 }, 
		{  -6,	6,	6, -6, -3, -3,	3,	3, -4,	4, -2,	2, -2, -2, -1, -1 }, 
		{	2,	0, -2,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	0 }, 
		{	0,	0,	0,	0,	2,	0, -2,	0,	0,	0,	0,	0,	1,	0,	1,	0 },
		{  -6,	6,	6, -6, -4, -2,	4,	2, -3,	3, -3,	3, -2, -1, -2, -1 },
		{	4, -4, -4,	4,	2,	2, -2, -2,	2, -2,	2, -2,	1,	1,	1,	1 }};


__constant__ real_t BSplineCP[4][4] = {
		 {  71 / 56.0, -19 / 56.0,   5 / 56.0,  -1 / 56.0 }, 
		 { -19 / 56.0,  95 / 56.0, -25 / 56.0,   5 / 56.0 }, 
		 {   5 / 56.0, -25 / 56.0,  95 / 56.0, -19 / 56.0 },
		 {  -1 / 56.0,   5 / 56.0, -19 / 56.0,  71 / 56.0 } 
	};
__constant__ real_t BSplineBase[4][4] = {
		{ -1 / 6.0,  3 / 6.0,  -3 / 6.0, 1 / 6.0 }, 
		{  3 / 6.0, -6 / 6.0,   3 / 6.0,       0 }, 
		{ -3 / 6.0,        0,   3 / 6.0,       0 }, 
		{  1 / 6.0,  4 / 6.0,   1 / 6.0,       0 } 
	};

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

/// \brief CUDA kernel to compute the 

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

/// \brief Kernel to compute Gradients of fImg on GPU, the results are stored in device
/// arrays. NOTE: The memory must be pre-allocated before calling this function. This kernel is 
/// especially useful for ICGN + Bicubic Bspline interpolation
///
/// \param fImg input image
/// \param iStartX x coordinate of the top-left point of ROI
/// \param iStartY y coordinate of the top-left point of ROI
/// \param iROIWidth Width of ROI
/// \param iROIHeight Height of ROI
/// \param iImgWidth Width of the image
/// \param iImgHeight Height of the image
/// \param Gx, Gy Gradient of fImgF in x,y dimensions
__global__ void Gradient_Kernel(// Inputs
								uchar1 *fImg,
							    int_t iStartX,   int_t iStartY,
							    int_t iROIWidth, int_t iROIHeight,
							    int_t iImgWidth, int_t iImgHeight,
							    // Outputs
							    real_t *Gx, real_t *Gy)
{
	// Block Index
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	
	// Thread Index
	const int ty = threadIdx.y;
	const int tx = threadIdx.x;
	
	// Global Memory offset: every block actually begin with 2 overlapped pixels
	const int y = iStartY - 1 + ty + (BLOCK_SIZE_Y - 2) * by;
	const int x = iStartX - 1 + tx + (BLOCK_SIZE_X - 2) * bx;

	// Declare the shared memory for storing the tiled subset
	__shared__ real_t img_sh[BLOCK_SIZE_Y][BLOCK_SIZE_X];

	// 1D index of the outpus
	int ind = 0;

	// Load the images into shared memory
	if (y < iStartY + iROIHeight + 1 && x < iStartX + iROIWidth + 1)
	{
		img_sh[ty][tx] = (real_t)fImg[y * iImgWidth + x].x;
	}
	__syncthreads();

	// Compute the gradients within the whole image, with 1-pixel shrinked on each boundary
	if (y >= iStartY && y < iROIHeight + iStartY && x >= iStartX && x < iROIWidth + iStartX && 
		tx != 0 && tx != BLOCK_SIZE_X - 1 && ty != 0 && ty != BLOCK_SIZE_Y - 1)
	{
		ind = (y - iStartY)*iROIWidth + (x - iStartX);
		Gx[ind] = 0.5 * (img_sh[ty][tx + 1] - img_sh[ty][tx - 1]);
		Gy[ind] = 0.5 * (img_sh[ty + 1][tx] - img_sh[ty - 1][tx]);
	}
}

/// \brief Kernel to compute Gradients of fImgF & fImgG on GPU, the results are stored in device
/// arrays. NOTE: The memory must be pre-allocated before calling this function. This kernel is 
/// especially useful for ICGN + Bicubic interpolation
///
/// \param fImgF Reference Image 
/// \param fImgG Target Image
/// \param iStartX x coordinate of the top-left point of ROI
/// \param iStartY y coordinate of the top-left point of ROI
/// \param iROIWidth Width of ROI
/// \param iROIHeight Height of ROI
/// \param iImgWidth Width of the image
/// \param iImgHeight Height of the image
/// \param Fx, Fy Gradient of fImgF in x,y dimensions
/// \param Gx, Gy, Gxy Gradient of fImgG in x, y adn xy dimensions
__global__ void GradientXY_2ImagesO2_Kernel(// Inputs
											uchar1 *fImgF, uchar1 *fImgG,
										    int_t iStartX,   int_t iStartY,
										    int_t iROIWidth, int_t iROIHeight,
										    int_t iImgWidth, int_t iImgHeight,
										    // Outputs
										    real_t *Fx, real_t *Fy,
										    real_t *Gx, real_t *Gy, real_t *Gxy)
{
	// Block Index
	const int by = blockIdx.y;
	const int bx = blockIdx.x;
	
	// Thread Index
	const int ty = threadIdx.y;
	const int tx = threadIdx.x;
	
	// Global Memory offset: every block actually begin with 2 overlapped pixels
	const int y = iStartY - 1 + ty + (BLOCK_SIZE_Y - 2) * by;
	const int x = iStartX - 1 + tx + (BLOCK_SIZE_X - 2) * bx;

	// Declare the shared memory for storing the tiled subset
	__shared__ real_t imgF_sh[BLOCK_SIZE_Y][BLOCK_SIZE_X];
	__shared__ real_t imgG_sh[BLOCK_SIZE_Y][BLOCK_SIZE_X];

	// 1D index of the outpus
	int ind = 0;

	// Load the images into shared memory
	if (y < iStartY + iROIHeight + 1 && x < iStartX + iROIWidth + 1)
	{
		imgF_sh[ty][tx] = (real_t)fImgF[y * iImgWidth + x].x;
		imgG_sh[ty][tx] = (real_t)fImgG[y * iImgWidth + x].x;
	}
	__syncthreads();

	// Compute the gradients within the whole image, with 1-pixel shrinked on each boundary
	if (y >= iStartY && y < iROIHeight + iStartY && x >= iStartX && x < iROIWidth + iStartX && 
		tx != 0 && tx != BLOCK_SIZE_X - 1 && ty != 0 && ty != BLOCK_SIZE_Y - 1)
	{
		ind = (y - iStartY)*iROIWidth + (x - iStartX);
		Fx[ind] = 0.5 * (imgF_sh[ty][tx + 1] - imgF_sh[ty][tx - 1]);
		Fy[ind] = 0.5 * (imgF_sh[ty + 1][tx] - imgF_sh[ty - 1][tx]);

		Gx[ind] = 0.5 * (imgG_sh[ty][tx + 1] - imgG_sh[ty][tx - 1]);
		Gy[ind] = 0.5 * (imgG_sh[ty + 1][tx] - imgG_sh[ty - 1][tx]);
		Gxy[ind]= 0.25* (imgG_sh[ty + 1][tx + 1] - imgG_sh[ty - 1][tx + 1] - imgG_sh[ty + 1][tx - 1] + imgG_sh[ty - 1][tx - 1]);
	}
}


/// \brief Kernel to compute Bicubic Interpolation LUT of fImgF & fImgG on GPU, the results are stored in device
/// arrays. NOTE: The memory must be pre-allocated before calling this function. This kernel is 
/// especially useful for ICGN + Bicubic interpolation
///
/// \param dIn_fImgT Image
/// \param dIn_fTx, dIn_fTy & dIn_fTxy Gradients of the image
/// \param iStartX x coordinate of the top-left point of ROI
/// \param iStartY y coordinate of the top-left point of ROI
/// \param iROIWidth Width of ROI
/// \param iROIHeight Height of ROI
/// \param iImgWidth Width of the image
/// \param iImgHeight Height of the image
/// \param dOut_fBicubicInterpolants LUT iROIHeight * iROIWidth * 4 * float4
__global__ void Bicubic_Kernel(// Inputs
				  			    const uchar1* dIn_fImgT, 
								const real_t* dIn_fTx, 
								const real_t* dIn_fTy, 
								const real_t* dIn_fTxy, 
								const int_t iStartX, const int_t iStartY,
								const int_t iROIWidth, const int_t iROIHeight, 
								const int_t iImgWidth, const int_t iImgHeight, 
								// Outputs
							    real_t4* dOut_fBicubicInterpolants)
{	
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	const int x = threadIdx.x + blockDim.x * blockIdx.x;

	// These two temporary arrays may consult 32 registers, 
	// half of the allowed ones for each thread's
	float fAlphaT[16], fTaoT[16];

	if (y < iROIHeight - 1 && x < iROIWidth - 1)
	{
		fTaoT[0] = (real_t)dIn_fImgT[(y + iStartY)*iImgWidth + iStartX + x].x;
		fTaoT[1] = (real_t)dIn_fImgT[(y + iStartY)*iImgWidth + iStartX + x + 1].x;
		fTaoT[2] = (real_t)dIn_fImgT[(y + 1 + iStartY)*iImgWidth + iStartX + x].x;
		fTaoT[3] = (real_t)dIn_fImgT[(y + 1 + iStartY)*iImgWidth + iStartX + x + 1].x;
		fTaoT[4] = dIn_fTx[y*iROIWidth + x];
		fTaoT[5] = dIn_fTx[y*iROIWidth + x + 1];
		fTaoT[6] = dIn_fTx[(y + 1)*iROIWidth + x];
		fTaoT[7] = dIn_fTx[(y + 1)*iROIWidth + x + 1];
		fTaoT[8] = dIn_fTy[y*iROIWidth + x];
		fTaoT[9] = dIn_fTy[y*iROIWidth + x + 1];
		fTaoT[10] = dIn_fTy[(y + 1)*iROIWidth + x];
		fTaoT[11] = dIn_fTy[(y + 1)*iROIWidth + x + 1];
		fTaoT[12] = dIn_fTxy[y*iROIWidth + x];
		fTaoT[13] = dIn_fTxy[y*iROIWidth + x + 1];
		fTaoT[14] = dIn_fTxy[(y + 1)*iROIWidth + x];
		fTaoT[15] = dIn_fTxy[(y + 1)*iROIWidth + x + 1];

		//Reduction to calculate fAlphaT (unroll the "for" loop)
		fAlphaT[0] = c_dBicubicMatrix[0][0] * fTaoT[0] + c_dBicubicMatrix[0][1] * fTaoT[1] + c_dBicubicMatrix[0][2] * fTaoT[2] + c_dBicubicMatrix[0][3] * fTaoT[3] +
			c_dBicubicMatrix[0][4] * fTaoT[4] + c_dBicubicMatrix[0][5] * fTaoT[5] + c_dBicubicMatrix[0][6] * fTaoT[6] + c_dBicubicMatrix[0][7] * fTaoT[7] +
			c_dBicubicMatrix[0][8] * fTaoT[8] + c_dBicubicMatrix[0][9] * fTaoT[9] + c_dBicubicMatrix[0][10] * fTaoT[10] + c_dBicubicMatrix[0][11] * fTaoT[11] +
			c_dBicubicMatrix[0][12] * fTaoT[12] + c_dBicubicMatrix[0][13] * fTaoT[13] + c_dBicubicMatrix[0][14] * fTaoT[14] + c_dBicubicMatrix[0][15] * fTaoT[15];
		fAlphaT[1] = c_dBicubicMatrix[1][0] * fTaoT[0] + c_dBicubicMatrix[1][1] * fTaoT[1] + c_dBicubicMatrix[1][2] * fTaoT[2] + c_dBicubicMatrix[1][3] * fTaoT[3] +
			c_dBicubicMatrix[1][4] * fTaoT[4] + c_dBicubicMatrix[1][5] * fTaoT[5] + c_dBicubicMatrix[1][6] * fTaoT[6] + c_dBicubicMatrix[1][7] * fTaoT[7] +
			c_dBicubicMatrix[1][8] * fTaoT[8] + c_dBicubicMatrix[1][9] * fTaoT[9] + c_dBicubicMatrix[1][10] * fTaoT[10] + c_dBicubicMatrix[1][11] * fTaoT[11] +
			c_dBicubicMatrix[1][12] * fTaoT[12] + c_dBicubicMatrix[1][13] * fTaoT[13] + c_dBicubicMatrix[1][14] * fTaoT[14] + c_dBicubicMatrix[1][15] * fTaoT[15];
		fAlphaT[2] = c_dBicubicMatrix[2][0] * fTaoT[0] + c_dBicubicMatrix[2][1] * fTaoT[1] + c_dBicubicMatrix[2][2] * fTaoT[2] + c_dBicubicMatrix[2][3] * fTaoT[3] +
			c_dBicubicMatrix[2][4] * fTaoT[4] + c_dBicubicMatrix[2][5] * fTaoT[5] + c_dBicubicMatrix[2][6] * fTaoT[6] + c_dBicubicMatrix[2][7] * fTaoT[7] +
			c_dBicubicMatrix[2][8] * fTaoT[8] + c_dBicubicMatrix[2][9] * fTaoT[9] + c_dBicubicMatrix[2][10] * fTaoT[10] + c_dBicubicMatrix[2][11] * fTaoT[11] +
			c_dBicubicMatrix[2][12] * fTaoT[12] + c_dBicubicMatrix[2][13] * fTaoT[13] + c_dBicubicMatrix[2][14] * fTaoT[14] + c_dBicubicMatrix[2][15] * fTaoT[15];
		fAlphaT[3] = c_dBicubicMatrix[3][0] * fTaoT[0] + c_dBicubicMatrix[3][1] * fTaoT[1] + c_dBicubicMatrix[3][2] * fTaoT[2] + c_dBicubicMatrix[3][3] * fTaoT[3] +
			c_dBicubicMatrix[3][4] * fTaoT[4] + c_dBicubicMatrix[3][5] * fTaoT[5] + c_dBicubicMatrix[3][6] * fTaoT[6] + c_dBicubicMatrix[3][7] * fTaoT[7] +
			c_dBicubicMatrix[3][8] * fTaoT[8] + c_dBicubicMatrix[3][9] * fTaoT[9] + c_dBicubicMatrix[3][10] * fTaoT[10] + c_dBicubicMatrix[3][11] * fTaoT[11] +
			c_dBicubicMatrix[3][12] * fTaoT[12] + c_dBicubicMatrix[3][13] * fTaoT[13] + c_dBicubicMatrix[3][14] * fTaoT[14] + c_dBicubicMatrix[3][15] * fTaoT[15];
		fAlphaT[4] = c_dBicubicMatrix[4][0] * fTaoT[0] + c_dBicubicMatrix[4][1] * fTaoT[1] + c_dBicubicMatrix[4][2] * fTaoT[2] + c_dBicubicMatrix[4][3] * fTaoT[3] +
			c_dBicubicMatrix[4][4] * fTaoT[4] + c_dBicubicMatrix[4][5] * fTaoT[5] + c_dBicubicMatrix[4][6] * fTaoT[6] + c_dBicubicMatrix[4][7] * fTaoT[7] +
			c_dBicubicMatrix[4][8] * fTaoT[8] + c_dBicubicMatrix[4][9] * fTaoT[9] + c_dBicubicMatrix[4][10] * fTaoT[10] + c_dBicubicMatrix[4][11] * fTaoT[11] +
			c_dBicubicMatrix[4][12] * fTaoT[12] + c_dBicubicMatrix[4][13] * fTaoT[13] + c_dBicubicMatrix[4][14] * fTaoT[14] + c_dBicubicMatrix[4][15] * fTaoT[15];
		fAlphaT[5] = c_dBicubicMatrix[5][0] * fTaoT[0] + c_dBicubicMatrix[5][1] * fTaoT[1] + c_dBicubicMatrix[5][2] * fTaoT[2] + c_dBicubicMatrix[5][3] * fTaoT[3] +
			c_dBicubicMatrix[5][4] * fTaoT[4] + c_dBicubicMatrix[5][5] * fTaoT[5] + c_dBicubicMatrix[5][6] * fTaoT[6] + c_dBicubicMatrix[5][7] * fTaoT[7] +
			c_dBicubicMatrix[5][8] * fTaoT[8] + c_dBicubicMatrix[5][9] * fTaoT[9] + c_dBicubicMatrix[5][10] * fTaoT[10] + c_dBicubicMatrix[5][11] * fTaoT[11] +
			c_dBicubicMatrix[5][12] * fTaoT[12] + c_dBicubicMatrix[5][13] * fTaoT[13] + c_dBicubicMatrix[5][14] * fTaoT[14] + c_dBicubicMatrix[5][15] * fTaoT[15];
		fAlphaT[6] = c_dBicubicMatrix[6][0] * fTaoT[0] + c_dBicubicMatrix[6][1] * fTaoT[1] + c_dBicubicMatrix[6][2] * fTaoT[2] + c_dBicubicMatrix[6][3] * fTaoT[3] +
			c_dBicubicMatrix[6][4] * fTaoT[4] + c_dBicubicMatrix[6][5] * fTaoT[5] + c_dBicubicMatrix[6][6] * fTaoT[6] + c_dBicubicMatrix[6][7] * fTaoT[7] +
			c_dBicubicMatrix[6][8] * fTaoT[8] + c_dBicubicMatrix[6][9] * fTaoT[9] + c_dBicubicMatrix[6][10] * fTaoT[10] + c_dBicubicMatrix[6][11] * fTaoT[11] +
			c_dBicubicMatrix[6][12] * fTaoT[12] + c_dBicubicMatrix[6][13] * fTaoT[13] + c_dBicubicMatrix[6][14] * fTaoT[14] + c_dBicubicMatrix[6][15] * fTaoT[15];
		fAlphaT[7] = c_dBicubicMatrix[7][0] * fTaoT[0] + c_dBicubicMatrix[7][1] * fTaoT[1] + c_dBicubicMatrix[7][2] * fTaoT[2] + c_dBicubicMatrix[7][3] * fTaoT[3] +
			c_dBicubicMatrix[7][4] * fTaoT[4] + c_dBicubicMatrix[7][5] * fTaoT[5] + c_dBicubicMatrix[7][6] * fTaoT[6] + c_dBicubicMatrix[7][7] * fTaoT[7] +
			c_dBicubicMatrix[7][8] * fTaoT[8] + c_dBicubicMatrix[7][9] * fTaoT[9] + c_dBicubicMatrix[7][10] * fTaoT[10] + c_dBicubicMatrix[7][11] * fTaoT[11] +
			c_dBicubicMatrix[7][12] * fTaoT[12] + c_dBicubicMatrix[7][13] * fTaoT[13] + c_dBicubicMatrix[7][14] * fTaoT[14] + c_dBicubicMatrix[7][15] * fTaoT[15];
		fAlphaT[8] = c_dBicubicMatrix[8][0] * fTaoT[0] + c_dBicubicMatrix[8][1] * fTaoT[1] + c_dBicubicMatrix[8][2] * fTaoT[2] + c_dBicubicMatrix[8][3] * fTaoT[3] +
			c_dBicubicMatrix[8][4] * fTaoT[4] + c_dBicubicMatrix[8][5] * fTaoT[5] + c_dBicubicMatrix[8][6] * fTaoT[6] + c_dBicubicMatrix[8][7] * fTaoT[7] +
			c_dBicubicMatrix[8][8] * fTaoT[8] + c_dBicubicMatrix[8][9] * fTaoT[9] + c_dBicubicMatrix[8][10] * fTaoT[10] + c_dBicubicMatrix[8][11] * fTaoT[11] +
			c_dBicubicMatrix[8][12] * fTaoT[12] + c_dBicubicMatrix[8][13] * fTaoT[13] + c_dBicubicMatrix[8][14] * fTaoT[14] + c_dBicubicMatrix[8][15] * fTaoT[15];
		fAlphaT[9] = c_dBicubicMatrix[9][0] * fTaoT[0] + c_dBicubicMatrix[9][1] * fTaoT[1] + c_dBicubicMatrix[9][2] * fTaoT[2] + c_dBicubicMatrix[9][3] * fTaoT[3] +
			c_dBicubicMatrix[9][4] * fTaoT[4] + c_dBicubicMatrix[9][5] * fTaoT[5] + c_dBicubicMatrix[9][6] * fTaoT[6] + c_dBicubicMatrix[9][7] * fTaoT[7] +
			c_dBicubicMatrix[9][8] * fTaoT[8] + c_dBicubicMatrix[9][9] * fTaoT[9] + c_dBicubicMatrix[9][10] * fTaoT[10] + c_dBicubicMatrix[9][11] * fTaoT[11] +
			c_dBicubicMatrix[9][12] * fTaoT[12] + c_dBicubicMatrix[9][13] * fTaoT[13] + c_dBicubicMatrix[9][14] * fTaoT[14] + c_dBicubicMatrix[9][15] * fTaoT[15];
		fAlphaT[10] = c_dBicubicMatrix[10][0] * fTaoT[0] + c_dBicubicMatrix[10][1] * fTaoT[1] + c_dBicubicMatrix[10][2] * fTaoT[2] + c_dBicubicMatrix[10][3] * fTaoT[3] +
			c_dBicubicMatrix[10][4] * fTaoT[4] + c_dBicubicMatrix[10][5] * fTaoT[5] + c_dBicubicMatrix[10][6] * fTaoT[6] + c_dBicubicMatrix[10][7] * fTaoT[7] +
			c_dBicubicMatrix[10][8] * fTaoT[8] + c_dBicubicMatrix[10][9] * fTaoT[9] + c_dBicubicMatrix[10][10] * fTaoT[10] + c_dBicubicMatrix[10][11] * fTaoT[11] +
			c_dBicubicMatrix[10][12] * fTaoT[12] + c_dBicubicMatrix[10][13] * fTaoT[13] + c_dBicubicMatrix[10][14] * fTaoT[14] + c_dBicubicMatrix[10][15] * fTaoT[15];
		fAlphaT[11] = c_dBicubicMatrix[11][0] * fTaoT[0] + c_dBicubicMatrix[11][1] * fTaoT[1] + c_dBicubicMatrix[11][2] * fTaoT[2] + c_dBicubicMatrix[11][3] * fTaoT[3] +
			c_dBicubicMatrix[11][4] * fTaoT[4] + c_dBicubicMatrix[11][5] * fTaoT[5] + c_dBicubicMatrix[11][6] * fTaoT[6] + c_dBicubicMatrix[11][7] * fTaoT[7] +
			c_dBicubicMatrix[11][8] * fTaoT[8] + c_dBicubicMatrix[11][9] * fTaoT[9] + c_dBicubicMatrix[11][10] * fTaoT[10] + c_dBicubicMatrix[11][11] * fTaoT[11] +
			c_dBicubicMatrix[11][12] * fTaoT[12] + c_dBicubicMatrix[11][13] * fTaoT[13] + c_dBicubicMatrix[11][14] * fTaoT[14] + c_dBicubicMatrix[11][15] * fTaoT[15];
		fAlphaT[12] = c_dBicubicMatrix[12][0] * fTaoT[0] + c_dBicubicMatrix[12][1] * fTaoT[1] + c_dBicubicMatrix[12][2] * fTaoT[2] + c_dBicubicMatrix[12][3] * fTaoT[3] +
			c_dBicubicMatrix[12][4] * fTaoT[4] + c_dBicubicMatrix[12][5] * fTaoT[5] + c_dBicubicMatrix[12][6] * fTaoT[6] + c_dBicubicMatrix[12][7] * fTaoT[7] +
			c_dBicubicMatrix[12][8] * fTaoT[8] + c_dBicubicMatrix[12][9] * fTaoT[9] + c_dBicubicMatrix[12][10] * fTaoT[10] + c_dBicubicMatrix[12][11] * fTaoT[11] +
			c_dBicubicMatrix[12][12] * fTaoT[12] + c_dBicubicMatrix[12][13] * fTaoT[13] + c_dBicubicMatrix[12][14] * fTaoT[14] + c_dBicubicMatrix[12][15] * fTaoT[15];
		fAlphaT[13] = c_dBicubicMatrix[13][0] * fTaoT[0] + c_dBicubicMatrix[13][1] * fTaoT[1] + c_dBicubicMatrix[13][2] * fTaoT[2] + c_dBicubicMatrix[13][3] * fTaoT[3] +
			c_dBicubicMatrix[13][4] * fTaoT[4] + c_dBicubicMatrix[13][5] * fTaoT[5] + c_dBicubicMatrix[13][6] * fTaoT[6] + c_dBicubicMatrix[13][7] * fTaoT[7] +
			c_dBicubicMatrix[13][8] * fTaoT[8] + c_dBicubicMatrix[13][9] * fTaoT[9] + c_dBicubicMatrix[13][10] * fTaoT[10] + c_dBicubicMatrix[13][11] * fTaoT[11] +
			c_dBicubicMatrix[13][12] * fTaoT[12] + c_dBicubicMatrix[13][13] * fTaoT[13] + c_dBicubicMatrix[13][14] * fTaoT[14] + c_dBicubicMatrix[13][15] * fTaoT[15];
		fAlphaT[14] = c_dBicubicMatrix[14][0] * fTaoT[0] + c_dBicubicMatrix[14][1] * fTaoT[1] + c_dBicubicMatrix[14][2] * fTaoT[2] + c_dBicubicMatrix[14][3] * fTaoT[3] +
			c_dBicubicMatrix[14][4] * fTaoT[4] + c_dBicubicMatrix[14][5] * fTaoT[5] + c_dBicubicMatrix[14][6] * fTaoT[6] + c_dBicubicMatrix[14][7] * fTaoT[7] +
			c_dBicubicMatrix[14][8] * fTaoT[8] + c_dBicubicMatrix[14][9] * fTaoT[9] + c_dBicubicMatrix[14][10] * fTaoT[10] + c_dBicubicMatrix[14][11] * fTaoT[11] +
			c_dBicubicMatrix[14][12] * fTaoT[12] + c_dBicubicMatrix[14][13] * fTaoT[13] + c_dBicubicMatrix[14][14] * fTaoT[14] + c_dBicubicMatrix[14][15] * fTaoT[15];
		fAlphaT[15] = c_dBicubicMatrix[15][0] * fTaoT[0] + c_dBicubicMatrix[15][1] * fTaoT[1] + c_dBicubicMatrix[15][2] * fTaoT[2] + c_dBicubicMatrix[15][3] * fTaoT[3] +
			c_dBicubicMatrix[15][4] * fTaoT[4] + c_dBicubicMatrix[15][5] * fTaoT[5] + c_dBicubicMatrix[15][6] * fTaoT[6] + c_dBicubicMatrix[15][7] * fTaoT[7] +
			c_dBicubicMatrix[15][8] * fTaoT[8] + c_dBicubicMatrix[15][9] * fTaoT[9] + c_dBicubicMatrix[15][10] * fTaoT[10] + c_dBicubicMatrix[15][11] * fTaoT[11] +
			c_dBicubicMatrix[15][12] * fTaoT[12] + c_dBicubicMatrix[15][13] * fTaoT[13] + c_dBicubicMatrix[15][14] * fTaoT[14] + c_dBicubicMatrix[15][15] * fTaoT[15];

		//Write the results back to the fBicubicInterpolants array
		dOut_fBicubicInterpolants[0 * iROIWidth*iROIHeight + (y*iROIWidth + x)].w = fAlphaT[0];
		dOut_fBicubicInterpolants[0 * iROIWidth*iROIHeight + (y*iROIWidth + x)].x = fAlphaT[1];
		dOut_fBicubicInterpolants[0 * iROIWidth*iROIHeight + (y*iROIWidth + x)].y = fAlphaT[2];
		dOut_fBicubicInterpolants[0 * iROIWidth*iROIHeight + (y*iROIWidth + x)].z = fAlphaT[3];
		dOut_fBicubicInterpolants[1 * iROIWidth*iROIHeight + (y*iROIWidth + x)].w = fAlphaT[4];
		dOut_fBicubicInterpolants[1 * iROIWidth*iROIHeight + (y*iROIWidth + x)].x = fAlphaT[5];
		dOut_fBicubicInterpolants[1 * iROIWidth*iROIHeight + (y*iROIWidth + x)].y = fAlphaT[6];
		dOut_fBicubicInterpolants[1 * iROIWidth*iROIHeight + (y*iROIWidth + x)].z = fAlphaT[7];
		dOut_fBicubicInterpolants[2 * iROIWidth*iROIHeight + (y*iROIWidth + x)].w = fAlphaT[8];
		dOut_fBicubicInterpolants[2 * iROIWidth*iROIHeight + (y*iROIWidth + x)].x = fAlphaT[9];
		dOut_fBicubicInterpolants[2 * iROIWidth*iROIHeight + (y*iROIWidth + x)].y = fAlphaT[10];
		dOut_fBicubicInterpolants[2 * iROIWidth*iROIHeight + (y*iROIWidth + x)].z = fAlphaT[11];
		dOut_fBicubicInterpolants[3 * iROIWidth*iROIHeight + (y*iROIWidth + x)].w = fAlphaT[12];
		dOut_fBicubicInterpolants[3 * iROIWidth*iROIHeight + (y*iROIWidth + x)].x = fAlphaT[13];
		dOut_fBicubicInterpolants[3 * iROIWidth*iROIHeight + (y*iROIWidth + x)].y = fAlphaT[14];
		dOut_fBicubicInterpolants[3 * iROIWidth*iROIHeight + (y*iROIWidth + x)].z = fAlphaT[15];
	}
}

__global__ void BicubicSplineLUT_Kernel(const uchar1* d_Img,
										const int_t iImgWidth, const int_t iImgHeight,
										const int_t iStartX, const int_t iStartY,
										const int_t iROIWidth, const int_t iROIHeight,
										real_t4 *d_OutBicubicSplineLUT)
{
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	const int x = threadIdx.x + blockDim.x * blockIdx.x;

	real_t fOmega[4][4];
	real_t fBeta[4][4];

	if (y < iROIHeight  && x < iROIWidth)
	{
		for (int k = 0; k < 4; k++)
		{
			for (int l = 0; l < 4; l++)
			{
				fOmega[k][l] = (real_t)d_Img[(y + iStartY - 1 + k)*iImgWidth + (x + iStartX - 1 + l)].x;
			}
		}
	}
}

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

void cuGradient(// Inputs
				uchar1 *fImg,
				int_t iStartX,   int_t iStartY,
				int_t iROIWidth, int_t iROIHeight,
				int_t iImgWidth, int_t iImgHeight,
				AccuracyOrder accuracy,
				// Outputs
				real_t *Gx, real_t *Gy)
{
	int_t iMarginX = iImgWidth - (iStartX + iROIWidth) + 1;
	int_t iMarginY = iImgHeight -(iStartY + iROIHeight) + 1;

	switch (accuracy)
	{
	case TW::AccuracyOrder::Quadratic:
	{
		if (iStartX < 1 || iStartY < 1 || iMarginX < 1 || iMarginY < 1)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
			dim3 blocks((int)ceil((float)(iROIWidth + 2) / (BLOCK_SIZE_X - 2)),
				(int)ceil((float)(iROIHeight + 2) / (BLOCK_SIZE_Y - 2)));

			cudaFuncSetCacheConfig(Gradient_Kernel, cudaFuncCachePreferShared);

			Gradient_Kernel<<<blocks, threads >>>(fImg,
												  iStartX, iStartY,
												  iROIWidth, iROIHeight,		
												  iImgWidth, iImgHeight,
												  Gx, Gy);
			getLastCudaError("Error in calling Gradient_Kernel");
		}
		break;
	}

	case TW::AccuracyOrder::Quartic:
	{
		if (iStartX < 2 || iStartY < 2 || iMarginX < 2 || iMarginY < 2)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			// TODO
		}

		break;
	}

	case TW::AccuracyOrder::Octic:
	{	
		if (iStartX < 4 || iStartY < 4 || iMarginX < 4 || iMarginY < 4)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			// TODO
		}

		break;
	}
	default:
	{
		break;
	}
	}
}

void cuGradientXY_2Images(// Inputs
						  uchar1 *fImgF, uchar1 *fImgG,
						  int_t iStartX,   int_t iStartY,
						  int_t iROIWidth, int_t iROIHeight,
						  int_t iImgWidth, int_t iImgHeight,
						  AccuracyOrder accuracy,
						  // Outputs
						  real_t *Fx, real_t *Fy,
						  real_t *Gx, real_t *Gy, real_t *Gxy)
{
	int_t iMarginX = iImgWidth - (iStartX + iROIWidth) + 1;
	int_t iMarginY = iImgHeight -(iStartY + iROIHeight) + 1;

	switch (accuracy)
	{
	case TW::AccuracyOrder::Quadratic:
	{
		if (iStartX < 1 || iStartY < 1 || iMarginX < 1 || iMarginY < 1)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
			dim3 blocks((int)ceil((float)(iROIWidth + 2) / (BLOCK_SIZE_X - 2)),
				(int)ceil((float)(iROIHeight + 2) / (BLOCK_SIZE_Y - 2)));

			cudaFuncSetCacheConfig(GradientXY_2ImagesO2_Kernel, cudaFuncCachePreferShared);

			GradientXY_2ImagesO2_Kernel <<<blocks, threads >>>(fImgF, fImgG,
															   iStartX, iStartY,
															   iROIWidth, iROIHeight,		
															   iImgWidth, iImgHeight,
															   Fx, Fy,
															   Gx, Gy, Gxy);
			getLastCudaError("Error in calling GradientXY_2ImagesO2_Kernel");
		}
		break;
	}

	case TW::AccuracyOrder::Quartic:
	{
		if (iStartX < 2 || iStartY < 2 || iMarginX < 2 || iMarginY < 2)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			// TODO
		}

		break;
	}

	case TW::AccuracyOrder::Octic:
	{	
		if (iStartX < 4 || iStartY < 4 || iMarginX < 4 || iMarginY < 4)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			// TODO
		}
		break;
	}
	default:
	{
		break;
	}
	}
}

void cuBicubicCoefficients(// Inputs
				  		   const uchar1* dIn_fImgT, 
						   const real_t* dIn_fTx, 
						   const real_t* dIn_fTy, 
						   const real_t* dIn_fTxy, 
						   const int iStartX, const int iStartY,
						   const int iROIWidth, const int iROIHeight, 
						   const int iImgWidth, const int iImgHeight, 
						   // Outputs
						   real_t4* dOut_fBicubicInterpolants)
{
	dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 blocks((iROIWidth - 1) / BLOCK_SIZE_X + 1, (iROIHeight - 1) / BLOCK_SIZE_Y + 1);

	cudaFuncSetCacheConfig(Bicubic_Kernel, cudaFuncCachePreferL1);
	Bicubic_Kernel<<<blocks, threads>>>(dIn_fImgT, 
										dIn_fTx, 
										dIn_fTy, 
										dIn_fTxy, 
										iStartX, iStartY, 
										iROIWidth, iROIHeight, 
										iImgWidth, iImgHeight,
										dOut_fBicubicInterpolants);
	getLastCudaError("Error in calling GradientXY_2ImagesO2_Kernel");
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