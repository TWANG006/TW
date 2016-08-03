#include "cuda_utils.cuh"

#include <thrust\extrema.h>
#include <thrust\device_ptr.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "TW_utils.h"

//---------------------------------------------------------------------------------------!
//---------------------------------------------------------------------------------------!
//-------------------------------------CUDA Kernels--------------------------------------!

// 1 Kb color map: from blue to red
 __constant__ unsigned int texture_data[256] =
    {
		0xFF830000, 0xFF870000, 0xFF8B0000, 0xFF8F0000, 0xFF930000, 0xFF970000, 0xFF9B0000, 0xFF9F0000,
		0xFFA30000, 0xFFA70000, 0xFFAB0000, 0xFFAF0000, 0xFFB30000, 0xFFB70000, 0xFFBB0000, 0xFFBF0000,
		0xFFC30000, 0xFFC70000, 0xFFCB0000, 0xFFCF0000, 0xFFD30000, 0xFFD70000, 0xFFDB0000, 0xFFDF0000,
		0xFFE30000, 0xFFE70000, 0xFFEB0000, 0xFFEF0000, 0xFFF30000, 0xFFF70000, 0xFFFB0000, 0xFFFF0000,
		0xFFFF0400, 0xFFFF0800, 0xFFFF0C00, 0xFFFF1000, 0xFFFF1400, 0xFFFF1800, 0xFFFF1C00, 0xFFFF2000,
		0xFFFF2400, 0xFFFF2800, 0xFFFF2C00, 0xFFFF3000, 0xFFFF3400, 0xFFFF3800, 0xFFFF3C00, 0xFFFF4000,
		0xFFFF4400, 0xFFFF4800, 0xFFFF4C00, 0xFFFF5000, 0xFFFF5400, 0xFFFF5800, 0xFFFF5C00, 0xFFFF6000,
		0xFFFF6400, 0xFFFF6800, 0xFFFF6C00, 0xFFFF7000, 0xFFFF7400, 0xFFFF7800, 0xFFFF7C00, 0xFFFF8000,
		0xFFFF8300, 0xFFFF8700, 0xFFFF8B00, 0xFFFF8F00, 0xFFFF9300, 0xFFFF9700, 0xFFFF9B00, 0xFFFF9F00,
		0xFFFFA300, 0xFFFFA700, 0xFFFFAB00, 0xFFFFAF00, 0xFFFFB300, 0xFFFFB700, 0xFFFFBB00, 0xFFFFBF00,
		0xFFFFC300, 0xFFFFC700, 0xFFFFCB00, 0xFFFFCF00, 0xFFFFD300, 0xFFFFD700, 0xFFFFDB00, 0xFFFFDF00,
		0xFFFFE300, 0xFFFFE700, 0xFFFFEB00, 0xFFFFEF00, 0xFFFFF300, 0xFFFFF700, 0xFFFFFB00, 0xFFFFFF00,
		0xFFFBFF04, 0xFFF7FF08, 0xFFF3FF0C, 0xFFEFFF10, 0xFFEBFF14, 0xFFE7FF18, 0xFFE3FF1C, 0xFFDFFF20,
		0xFFDBFF24, 0xFFD7FF28, 0xFFD3FF2C, 0xFFCFFF30, 0xFFCBFF34, 0xFFC7FF38, 0xFFC3FF3C, 0xFFBFFF40,
		0xFFBBFF44, 0xFFB7FF48, 0xFFB3FF4C, 0xFFAFFF50, 0xFFABFF54, 0xFFA7FF58, 0xFFA3FF5C, 0xFF9FFF60,
		0xFF9BFF64, 0xFF97FF68, 0xFF93FF6C, 0xFF8FFF70, 0xFF8BFF74, 0xFF87FF78, 0xFF83FF7C, 0xFF80FF80,
		0xFF7CFF83, 0xFF78FF87, 0xFF74FF8B, 0xFF70FF8F, 0xFF6CFF93, 0xFF68FF97, 0xFF64FF9B, 0xFF60FF9F,
		0xFF5CFFA3, 0xFF58FFA7, 0xFF54FFAB, 0xFF50FFAF, 0xFF4CFFB3, 0xFF48FFB7, 0xFF44FFBB, 0xFF40FFBF,
		0xFF3CFFC3, 0xFF38FFC7, 0xFF34FFCB, 0xFF30FFCF, 0xFF2CFFD3, 0xFF28FFD7, 0xFF24FFDB, 0xFF20FFDF,
		0xFF1CFFE3, 0xFF18FFE7, 0xFF14FFEB, 0xFF10FFEF, 0xFF0CFFF3, 0xFF08FFF7, 0xFF04FFFB, 0xFF00FFFF,
		0xFF00FBFF, 0xFF00F7FF, 0xFF00F3FF, 0xFF00EFFF, 0xFF00EBFF, 0xFF00E7FF, 0xFF00E3FF, 0xFF00DFFF,
		0xFF00DBFF, 0xFF00D7FF, 0xFF00D3FF, 0xFF00CFFF, 0xFF00CBFF, 0xFF00C7FF, 0xFF00C3FF, 0xFF00BFFF,
		0xFF00BBFF, 0xFF00B7FF, 0xFF00B3FF, 0xFF00AFFF, 0xFF00ABFF, 0xFF00A7FF, 0xFF00A3FF, 0xFF009FFF,
		0xFF009BFF, 0xFF0097FF, 0xFF0093FF, 0xFF008FFF, 0xFF008BFF, 0xFF0087FF, 0xFF0083FF, 0xFF0080FF,
		0xFF007CFF, 0xFF0078FF, 0xFF0074FF, 0xFF0070FF, 0xFF006CFF, 0xFF0068FF, 0xFF0064FF, 0xFF0060FF,
		0xFF005CFF, 0xFF0058FF, 0xFF0054FF, 0xFF0050FF, 0xFF004CFF, 0xFF0048FF, 0xFF0044FF, 0xFF0040FF,
		0xFF003CFF, 0xFF0038FF, 0xFF0034FF, 0xFF0030FF, 0xFF002CFF, 0xFF0028FF, 0xFF0024FF, 0xFF0020FF,
		0xFF001CFF, 0xFF0018FF, 0xFF0014FF, 0xFF0010FF, 0xFF000CFF, 0xFF0008FF, 0xFF0004FF, 0xFF0000FF,
		0xFF0000FB, 0xFF0000F7, 0xFF0000F3, 0xFF0000EF, 0xFF0000EB, 0xFF0000E7, 0xFF0000E3, 0xFF0000DF,
		0xFF0000DB, 0xFF0000D7, 0xFF0000D3, 0xFF0000CF, 0xFF0000CB, 0xFF0000C7, 0xFF0000C3, 0xFF0000BF,
		0xFF0000BB, 0xFF0000B7, 0xFF0000B3, 0xFF0000AF, 0xFF0000AB, 0xFF0000A7, 0xFF0000A3, 0xFF00009F,
		0xFF00009B, 0xFF000097, 0xFF000093, 0xFF00008F, 0xFF00008B, 0xFF000087, 0xFF000083, 0xFF000080
    };


/*Problems to think about: 
  1. Use the whole subset or only the POI?
*/
__global__ void constructTextImage_Kernel(// Outputs
										  unsigned int* texImgU,
										  unsigned int* texImgV,
										  // Inputs
										  int *iPOIpos,
										  TW::real_t* fU,
										  TW::real_t* fV,
										  int iNumPOI,
										  int iStartX, int iStartY,
										  int iROIWidth, int iROIHeight,
										  TW::real_t *fMaxU, TW::real_t *fMinU,
										  TW::real_t *fMaxV, TW::real_t *fMinV)
{
	TW::real_t tempU, tempV;
	int tempIndOthers[9];

	for (auto i = blockIdx.x * blockDim.x + threadIdx.x;
			  i < iNumPOI;
			  i += blockDim.x*gridDim.x)
	{
		tempU = fU[i] - fMinU[0];
		tempV = fV[i] - fMinV[0];

		tempIndOthers[0] = (iPOIpos[i * 2 + 0] - iStartY + 1) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX - 1;
		tempIndOthers[1] = (iPOIpos[i * 2 + 0] - iStartY + 1) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX;
		tempIndOthers[2] = (iPOIpos[i * 2 + 0] - iStartY + 1) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX + 1;
		tempIndOthers[3] = (iPOIpos[i * 2 + 0] - iStartY) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX - 1;
		tempIndOthers[4] = (iPOIpos[i * 2 + 0] - iStartY) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX;
		tempIndOthers[5] = (iPOIpos[i * 2 + 0] - iStartY) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX + 1;
		tempIndOthers[6] = (iPOIpos[i * 2 + 0] - iStartY - 1) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX - 1;
		tempIndOthers[7] = (iPOIpos[i * 2 + 0] - iStartY - 1) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX;
		tempIndOthers[7] = (iPOIpos[i * 2 + 0] - iStartY - 1) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX + 1;

		texImgU[tempIndOthers[0]] = texture_data[int(255 * tempU / (fMaxU[0] - fMinU[0]))];
		texImgV[tempIndOthers[0]] = texture_data[int(255 * tempV / (fMaxV[0] - fMinV[0]))];
		texImgU[tempIndOthers[1]] = texture_data[int(255 * tempU / (fMaxU[0] - fMinU[0]))];
		texImgV[tempIndOthers[1]] = texture_data[int(255 * tempV / (fMaxV[0] - fMinV[0]))];
		texImgU[tempIndOthers[2]] = texture_data[int(255 * tempU / (fMaxU[0] - fMinU[0]))];
		texImgV[tempIndOthers[2]] = texture_data[int(255 * tempV / (fMaxV[0] - fMinV[0]))];
		texImgU[tempIndOthers[3]] = texture_data[int(255 * tempU / (fMaxU[0] - fMinU[0]))];
		texImgV[tempIndOthers[3]] = texture_data[int(255 * tempV / (fMaxV[0] - fMinV[0]))];
		texImgU[tempIndOthers[4]] = texture_data[int(255 * tempU / (fMaxU[0] - fMinU[0]))];
		texImgV[tempIndOthers[4]] = texture_data[int(255 * tempV / (fMaxV[0] - fMinV[0]))];
		texImgU[tempIndOthers[5]] = texture_data[int(255 * tempU / (fMaxU[0] - fMinU[0]))];
		texImgV[tempIndOthers[5]] = texture_data[int(255 * tempV / (fMaxV[0] - fMinV[0]))];
		texImgU[tempIndOthers[6]] = texture_data[int(255 * tempU / (fMaxU[0] - fMinU[0]))];
		texImgV[tempIndOthers[6]] = texture_data[int(255 * tempV / (fMaxV[0] - fMinV[0]))];
		texImgU[tempIndOthers[7]] = texture_data[int(255 * tempU / (fMaxU[0] - fMinU[0]))];
		texImgV[tempIndOthers[7]] = texture_data[int(255 * tempV / (fMaxV[0] - fMinV[0]))];
		texImgU[tempIndOthers[8]] = texture_data[int(255 * tempU / (fMaxU[0] - fMinU[0]))];
		texImgV[tempIndOthers[8]] = texture_data[int(255 * tempV / (fMaxV[0] - fMinV[0]))];
	}
}

__global__ void constructTextImageFixedMinMax_Kernel(// Outputs
										  unsigned int* texImgU,
										  unsigned int* texImgV,
										  // Inputs
										  int *iPOIpos,
										  TW::real_t* fU,
										  TW::real_t* fV,
										  TW::real_t* fAccumulateU,
										  TW::real_t* fAccumulateV,
										  int iNumPOI,
										  int iStartX, int iStartY,
										  int iROIWidth, int iROIHeight,
										  TW::real_t fMaxU, TW::real_t fMinU,
										  TW::real_t fMaxV, TW::real_t fMinV)
{
	TW::real_t tempU, tempV;
	int tempIndOthers[9];

	for (auto i = blockIdx.x * blockDim.x + threadIdx.x;
			  i < iNumPOI;
			  i += blockDim.x*gridDim.x)
	{
		tempU = fU[i] + fAccumulateU[i];
		tempV = fV[i] + fAccumulateV[i];

		tempU > fMaxU? tempU = fMaxU:tempU;
		tempV > fMaxV? tempV = fMaxV:tempV;

		tempU < fMinU? tempU = fMinU:tempU;
		tempV < fMinV? tempV = fMinV:tempV;

		tempU = tempU - fMinU;
		tempV = tempV - fMinV;

		tempIndOthers[0] = (iPOIpos[i * 2 + 0] - iStartY + 1) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX - 1;
		tempIndOthers[1] = (iPOIpos[i * 2 + 0] - iStartY + 1) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX;
		tempIndOthers[2] = (iPOIpos[i * 2 + 0] - iStartY + 1) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX + 1;
		tempIndOthers[3] = (iPOIpos[i * 2 + 0] - iStartY) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX - 1;
		tempIndOthers[4] = (iPOIpos[i * 2 + 0] - iStartY) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX;
		tempIndOthers[5] = (iPOIpos[i * 2 + 0] - iStartY) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX + 1;
		tempIndOthers[6] = (iPOIpos[i * 2 + 0] - iStartY - 1) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX - 1;
		tempIndOthers[7] = (iPOIpos[i * 2 + 0] - iStartY - 1) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX;
		tempIndOthers[8] = (iPOIpos[i * 2 + 0] - iStartY - 1) * iROIWidth + iPOIpos[i * 2 + 1] - iStartX + 1;

		texImgU[tempIndOthers[0]] = texture_data[int(255 * tempU / (fMaxU- fMinU))];
		texImgV[tempIndOthers[0]] = texture_data[int(255 * tempV / (fMaxV - fMinV))];
		texImgU[tempIndOthers[1]] = texture_data[int(255 * tempU / (fMaxU - fMinU))];
		texImgV[tempIndOthers[1]] = texture_data[int(255 * tempV / (fMaxV - fMinV))];
		texImgU[tempIndOthers[2]] = texture_data[int(255 * tempU / (fMaxU - fMinU))];
		texImgV[tempIndOthers[2]] = texture_data[int(255 * tempV / (fMaxV - fMinV))];
		texImgU[tempIndOthers[3]] = texture_data[int(255 * tempU / (fMaxU - fMinU))];
		texImgV[tempIndOthers[3]] = texture_data[int(255 * tempV / (fMaxV - fMinV))];
		texImgU[tempIndOthers[4]] = texture_data[int(255 * tempU / (fMaxU - fMinU))];
		texImgV[tempIndOthers[4]] = texture_data[int(255 * tempV / (fMaxV - fMinV))];
		texImgU[tempIndOthers[5]] = texture_data[int(255 * tempU / (fMaxU - fMinU))];
		texImgV[tempIndOthers[5]] = texture_data[int(255 * tempV / (fMaxV - fMinV))];
		texImgU[tempIndOthers[6]] = texture_data[int(255 * tempU / (fMaxU - fMinU))];
		texImgV[tempIndOthers[6]] = texture_data[int(255 * tempV / (fMaxV - fMinV))];
		texImgU[tempIndOthers[7]] = texture_data[int(255 * tempU / (fMaxU - fMinU))];
		texImgV[tempIndOthers[7]] = texture_data[int(255 * tempV / (fMaxV - fMinV))];
		texImgU[tempIndOthers[8]] = texture_data[int(255 * tempU / (fMaxU - fMinU))];
		texImgV[tempIndOthers[8]] = texture_data[int(255 * tempV / (fMaxV - fMinV))];
	}
}

__global__ void updatePOIpos_Kernel(TW::real_t *fU,
									TW::real_t *fV,
									int iNumberX, int iNumberY,
									int *iPOIpos)
{
	int iPOINum = iNumberX * iNumberY;
	for (auto i = blockIdx.x * blockDim.x + threadIdx.x;
			  i < iPOINum;
			  i += blockDim.x * gridDim.x)
	{
		iPOIpos[i * 2 + 0] += int(fV[i]);
		iPOIpos[i * 2 + 1] += int(fU[i]);
	}
}

__global__ void accumulatePOI_Kernel(// Inputs
									TW::real_t *fU,
									TW::real_t *fV,
									int *iCurrentPOIXY,
									int iPOINum,
									// Outputs
									int *iPOIXY)
{
	for (auto i = blockIdx.x * blockDim.x + threadIdx.x;
			  i < iPOINum;
			  i += blockDim.x * gridDim.x)
	{
		iPOIXY[i * 2 + 0] = iCurrentPOIXY[i * 2 + 0] + int(fV[i]);
		iPOIXY[i * 2 + 1] = iCurrentPOIXY[i * 2 + 1] + int(fU[i]);
	}
}

__global__ void accumulatePOINew_Kernel(// Inputs
									TW::real_t *fU,
									TW::real_t *fV,
									TW::real_t *fAccumulateU,
									TW::real_t *fAccumulateV,
									int *iCurrentPOIXY,
									int iPOINum,
									// Outputs
									int *iPOIXY)
{
	for (auto i = blockIdx.x * blockDim.x + threadIdx.x;
			  i < iPOINum;
			  i += blockDim.x * gridDim.x)
	{
		iPOIXY[i * 2 + 0] = iCurrentPOIXY[i * 2 + 0] + int(fV[i]) - int(fAccumulateU[i]);
		iPOIXY[i * 2 + 1] = iCurrentPOIXY[i * 2 + 1] + int(fU[i]) - int(fAccumulateV[i]);
	}
}

__global__ void accumulateUV_Kernel(// Inputs
									TW::real_t *fCurrentU,
									TW::real_t *fCurrentV,
									int iPOINum,
									// Outputs
									TW::real_t *fU,
									TW::real_t *fV)
{
	for (auto i = blockIdx.x * blockDim.x + threadIdx.x;
			  i < iPOINum;
			  i += blockDim.x * gridDim.x)
	{
		fCurrentU[i] += fU[i];
		fCurrentV[i] += fV[i];
	}
}


//------------------------------------/CUDA Kernels--------------------------------------!
//---------------------------------------------------------------------------------------!
//---------------------------------------------------------------------------------------!

//---------------------------------------------------------------------------------------!
//---------------------------------------------------------------------------------------!
//---------------------------------------Wrappers----------------------------------------!
void minMaxRWrapper(TW::real_t *&iU, TW::real_t *&iV, int iNU, int iNV,
				    TW::real_t* &iminU, TW::real_t* &imaxU,
					TW::real_t* &iminV, TW::real_t* &imaxV)
{
	using iThDevPtr = thrust::device_ptr<TW::real_t>;

	// Use thrust to find max and min simultaneously
	iThDevPtr d_Uptr(iU);
	thrust::pair<iThDevPtr, iThDevPtr> result_u = thrust::minmax_element(d_Uptr, d_Uptr + iNU - 1);
	// Cast the thrust device pointer to raw device pointer
	iminU = thrust::raw_pointer_cast(result_u.first);
	imaxU = thrust::raw_pointer_cast(result_u.second);

	// Same for iV
	iThDevPtr d_Vptr(iV);
	thrust::pair<iThDevPtr, iThDevPtr> result_v = thrust::minmax_element(d_Vptr, d_Vptr + iNV - 1);
	// Cast the thrust device pointer to raw device pointer
	iminV = thrust::raw_pointer_cast(result_u.first);
	imaxV = thrust::raw_pointer_cast(result_u.second);
}

void cuUpdatePOIpos(// Inputs
					TW::real_t *fU,
				    TW::real_t *fV,
					int iNumberX, int iNumberY,
					// Outputs
					int *iPOIpos)
{
	//int numSMs;
	//cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

	updatePOIpos_Kernel<<<256, BLOCK_SIZE_64>>>(fU,
													   fV,
													   iNumberX, iNumberY,
													   iPOIpos);
}

void cuAccumulatePOI(// Inputs
						TW::real_t *fU,
						TW::real_t *fV,
						int *iCurrentPOIXY,
						int iNumPOI,
						// Outputs
						int *iPOIXY)
{
	//int numSMs;
	//cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

	accumulatePOI_Kernel<<<256, BLOCK_SIZE_64>>>(fU,
														   fV,
														   iCurrentPOIXY,
														   iNumPOI,
														   iPOIXY);
}

void cuAccumulatePOI(// Inputs
						TW::real_t *fU,
						TW::real_t *fV,
						TW::real_t *fAccumulateU,
						TW::real_t *fAccumulateV,
						int *iCurrentPOIXY,
						int iNumPOI,
						// Outputs
						int *iPOIXY)
{
	//int numSMs;
	//cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

	accumulatePOINew_Kernel<<<256, BLOCK_SIZE_64>>>(fU,
														   fV,
														   fAccumulateU,
														   fAccumulateV,
														   iCurrentPOIXY,
														   iNumPOI,
														   iPOIXY);
}

void cuAccumulateUV(// Inputs
									TW::real_t *fCurrentU,
									TW::real_t *fCurrentV,
									int iPOINum,
									// Outputs
									TW::real_t *fU,
									TW::real_t *fV)
{
	accumulateUV_Kernel<<<256, BLOCK_SIZE_64>>>(// Inputs
									fCurrentU,
									fCurrentV,
									iPOINum,
									// Outputs
									fU,
									fV);
}

void constructTextImage(// Outputs
						unsigned int* texImgU,
						unsigned int* texImgV,
						// Inputs
						int *iPOIpos,
						TW::real_t* fU,
						TW::real_t* fV,
						int iNumPOI,
						int iStartX, int iStartY,
						int iROIWidth, int iROIHeight,
						TW::real_t *fMaxU, TW::real_t *fMinU,
						TW::real_t *fMaxV, TW::real_t *fMinV)
{
	constructTextImage_Kernel<<<256, BLOCK_SIZE_64>>>(texImgU,
													  texImgV,
													  iPOIpos,
													  fU,
													  fV,
													  iNumPOI,
													  iStartX, iStartY,
													  iROIWidth, iROIHeight,
													  fMaxU, fMinU,
													  fMaxV, fMinV);
}

void constructTextImageFixedMinMax(// Outputs
						unsigned int* texImgU,
						unsigned int* texImgV,
						// Inputs
						int *iPOIpos,
						TW::real_t* fU,
						TW::real_t* fV,
						TW::real_t* fAccumulateU,
						TW::real_t* fAccumulateV,
						int iNumPOI,
						int iStartX, int iStartY,
						int iROIWidth, int iROIHeight,
						TW::real_t fMaxU, TW::real_t fMinU,
						TW::real_t fMaxV, TW::real_t fMinV)
{
	constructTextImageFixedMinMax_Kernel<<<256, BLOCK_SIZE_64>>>(texImgU,
													  texImgV,
													  iPOIpos,
													  fU,
													  fV,
													  fAccumulateU,
													  fAccumulateV,
													  iNumPOI,
													  iStartX, iStartY,
													  iROIWidth, iROIHeight,
													  fMaxU, fMinU,
													  fMaxV, fMinV);
}

//---------------------------------------------------------------------------------------!
//---------------------------------------------------------------------------------------!
//--------------------------------------/Wrappers----------------------------------------!