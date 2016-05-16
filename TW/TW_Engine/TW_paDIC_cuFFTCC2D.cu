#include "TW_paDIC_cuFFTCC2D.h"

#include <iostream>
#include <stdexcept>

#include "TW_MemManager.h"
#include "TW_utils.h"

#include <cuda.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"


namespace TW{
namespace paDIC{

// !-----------------------------CUDA Kernel Functions -----------------------------------
		
/// \brief Compute complex numbers' multiplication and scale for a whole set of subsets used 
/// in the parallel FFTCC algoirthm to ultimately acclerate the computation.
///
/// \param *w_a array which hold the FFT-transformed values for all the subsets in refImg
/// \param *w_b array which hold the FFT-transformed values for all the subsets in tarImg
/// \param m_iFFTSubH subset height of the FFT-CC subset
/// \param m_iFFTSubW subset width of the FFT-CC subset
/// \param m_dModf normalization parameter of refImg
/// \param m_dModg normalization parameter of tarImg
/// \param *w_c the result array
__global__ void complexMulandScale_kernel(// Inputs
										  const cudafftComplex *w_a, const cudafftComplex *w_b,
										  int_t m_iFFTSubH, int_t m_iFFTSubW,
										  real_t *m_dModf, real_t *m_dModg,
										  // Output
										  cudafftComplex*w_c)
{
	auto tid = threadIdx.x;
	auto bid = blockIdx.x;
	auto dim = blockDim.x;
	auto size = m_iFFTSubW * (m_iFFTSubH / 2 + 1);
	const cudafftComplex * a = w_a + bid*size;
	const cudafftComplex * b = w_b + bid*size;
	cudafftComplex * c = w_c + bid*size;

	for (auto i = tid; i < size; i += dim)
	{
		c[i] = ComplexScale(ComplexMul(a[i], b[i]), 
                            1.0 / (sqrt(m_dModf[bid] * m_dModg[bid]) * m_iFFTSubW * m_iFFTSubH));
	}
}

/// \brief Using block-level parallel reduction algorithm to compute the maximum ZNCC values for 
/// each subset in parallel. Each thread block is responsible for the calculation of one subset.
/// 
///
/// \param *w_Subset
__global__ void findMax(real_t*w_SubsetC,
						int_t m_iFFTSubH, int_t m_iFFTSubW,
						int_t m_iSubsetX, int_t m_iSubsetY,
						//return val
						real_t *m_fU, real_t *m_fV,
						real_t *m_dZNCC)
{
	auto tid = threadIdx.x;
	auto dim = blockDim.x;
	auto bid = blockIdx.x;
	__shared__ real_t sdata[BLOCK_SIZE_256];
	__shared__ int_t sind[BLOCK_SIZE_256];

	auto size = m_iFFTSubW * m_iFFTSubH;
	real_t *m_SubsetC = w_SubsetC + bid*(m_iFFTSubW * m_iFFTSubH);
	real_t data = m_SubsetC[tid];
	auto ind = tid;

	for (auto id = tid + dim; id<size; id += dim)
	{
		if (data<m_SubsetC[id])
		{
			data = m_SubsetC[id];
			ind = id;
		}
	}
	reduceToMaxBlock<BLOCK_SIZE_256, float>(sdata, sind, data, ind, tid);

	ind = sind[0];
	int_t peakx = ind%m_iFFTSubW;
	int_t peaky = ind / m_iFFTSubW;
	if (peakx>m_iSubsetX)
		peakx -= m_iFFTSubW;
	if (peaky>m_iSubsetY)
		peaky -= m_iFFTSubH;
	if (tid == 0)
	{
		m_fU[bid] = real_t(peakx);
		m_fV[bid] = real_t(peaky);
		m_dZNCC[bid] = sdata[0];
		//m_dZNCC[bid] = data;
	}
}


__global__ void cufft_prepare_kernel(// Inputs
									 int_t *m_dPXY,
									 uchar1 *m_dR, uchar1 *m_dT,
									 int_t m_iFFTSubH, int_t m_iFFTSubW,
									 int_t m_iSubsetX, int_t m_iSubsetY,
									 int_t m_iHeight, int_t m_iWidth,
									 // Outputs
									 real_t *w_Subset1, real_t * w_Subset2,
									 real_t *m_dMod1, real_t *m_dMod2)
{
	__shared__ float sm[BLOCK_SIZE_256];
	auto bid = blockIdx.x;
	auto dim = blockDim.x;
	auto tid = threadIdx.x;
	real_t d_tempt;
	real_t d_sumR, d_sumT;
	real_t d_aveR, d_aveT;
	d_sumR = 0;
	d_sumT = 0;
	real_t *m_Subset1 = w_Subset1 + bid*(m_iFFTSubW * m_iFFTSubH);
	real_t *m_Subset2 = w_Subset2 + bid*(m_iFFTSubW * m_iFFTSubH);
	auto size = m_iFFTSubH*m_iFFTSubW;

	for (auto id = tid; id<size; id += dim)
	{
		int_t l = id / m_iFFTSubW;
		int_t m = id%m_iFFTSubW;
		d_tempt = (real_t)m_dR[(int_t(m_dPXY[bid * 2] - m_iSubsetY + l))*m_iWidth + int_t(m_dPXY[bid * 2 + 1] - m_iSubsetX + m)].x;
		m_Subset1[id] = d_tempt;
		d_sumR += d_tempt / size;

		d_tempt = (real_t)m_dT[(int_t(m_dPXY[bid * 2] - m_iSubsetY + l))*m_iWidth + int_t(m_dPXY[bid * 2 + 1] - m_iSubsetX + m)].x;
		m_Subset2[id] = d_tempt;
		d_sumT += d_tempt / size;
	}

	/*d_aveR = blockReduceSum<BLOCK_SIZE_256, float>(d_sumR);
	d_aveT = blockReduceSum<BLOCK_SIZE_256, float>(d_sumT);*/

	reduceBlock<BLOCK_SIZE_256, real_t>(sm, d_sumR, tid);
	d_aveR = sm[0];
	__syncthreads();
	reduceBlock<BLOCK_SIZE_256, real_t>(sm, d_sumT, tid);
	d_aveT = sm[0];
	__syncthreads();
	
	d_sumR = 0;
	d_sumT = 0;
	
	for (auto id = tid; id<size; id += dim)
	{
		d_tempt = m_Subset1[id] - d_aveR;
		m_Subset1[id] = d_tempt;
		d_sumR += pow(d_tempt, 2);

		d_tempt = m_Subset2[id] - d_aveT;
		m_Subset2[id] = d_tempt;
		d_sumT += pow(d_tempt, 2);
	}

	reduceBlock<BLOCK_SIZE_256, float>(sm, d_sumR, tid);
	if (tid == 0)
		d_aveR = sm[0];
	reduceBlock<BLOCK_SIZE_256, float>(sm, d_sumT, tid);
	if (tid == 0)
		d_aveT = sm[0];

	if (tid == 0)
	{
		m_dMod1[bid] = d_aveR;
		m_dMod2[bid] = d_aveT;
	}
}


// ------------------------------CUDA Kernel Functions End-------------------------------!
		
cuFFTCC2D::cuFFTCC2D(const int_t iROIWidth,		 const int_t iROIHeight,
					 const int_t iSubsetX,		 const int_t iSubsetY,
					 const int_t iGridSpaceX,	 const int_t iGridSpaceY,
					 const int_t iMarginX,		 const int_t iMarginY)
	: Fftcc2D(iROIWidth, 	  iROIHeight,
			  iSubsetX,		  iSubsetY,
			  iGridSpaceX,	  iGridSpaceY,
			  iMarginX,		  iMarginY)
	, isLowLevelApiCalled(false)
	, isDestroyed(false)
{
	if (!recomputeNumPOI())
		throw std::logic_error("Number of POIs is below 0!");
}

cuFFTCC2D::cuFFTCC2D(const int_t iImgWidth,	  const int_t iImgHeight,
				     const int_t iROIWidth,   const int_t iROIHeight,
					 const int_t iStartX,     const int_t iStartY,
					 const int_t iSubsetX,    const int_t iSubsetY,
					 const int_t iGridSpaceX, const int_t iGridSpaceY,
					 const int_t iMarginX,    const int_t iMarginY)
	: Fftcc2D(iImgWidth,   iImgHeight,
			  iStartX,     iStartY,
			  iROIWidth,   iROIHeight,
			  iSubsetX,    iSubsetY,
			  iGridSpaceX, iGridSpaceY,
			  iMarginX,    iMarginY)
	, isLowLevelApiCalled(false)
	, isDestroyed(false)
{
	if (!recomputeNumPOI())
		throw std::logic_error("Number of POIs is below 0!");
}

cuFFTCC2D::~cuFFTCC2D()
{}


void cuFFTCC2D::InitializeFFTCC(// Output
								real_t**& fU,
								real_t**& fV,
								real_t**& fZNCC,
								// Input
								const cv::Mat& refImg)
{
	//!- Check if the low level api is called or not
	if (isLowLevelApiCalled)
	{
		std::cout << "The low-level GPU APIs are already initialized!\n";
		return;
	}

	//!- Precompute the POI postions, since this is invariant during the entire
	// computation.
	//!- Determine whether the whole image or the ROI is used
	if(!m_isWholeImgUsed)
		cuComputePOIPositions(m_cuHandle.m_d_iPOIXY,
							  m_iNumPOIX,    m_iNumPOIY,
							  m_iMarginX,    m_iMarginY,
							  m_iSubsetX,    m_iSubsetY,
							  m_iGridSpaceX, m_iGridSpaceY);
	else
		cuComputePOIPositions(m_cuHandle.m_d_iPOIXY,
							  m_iStartX,     m_iStartY,
							  m_iNumPOIX,    m_iNumPOIY,
							  m_iMarginX,    m_iMarginY,
							  m_iSubsetX,    m_iSubsetY,
							  m_iGridSpaceX, m_iGridSpaceY);

	int_t iPOINum = GetNumPOIs();

	//!- Allocate host memory
	hcreateptr<real_t>(fU, m_iNumPOIY, m_iNumPOIX);
	hcreateptr<real_t>(fV, m_iNumPOIY, m_iNumPOIX);
	hcreateptr<real_t>(fZNCC, m_iNumPOIY, m_iNumPOIX);
	
	int_t iROISize = GetROISize();
	int_t iFFTSubW = m_iSubsetX * 2, iFFTSubH = m_iSubsetY * 2;
	int_t iFFTSize = iFFTSubW * iFFTSubH;
	int_t iFFTFreqSize = iFFTSubW * (iFFTSubH / 2 + 1);

	//!- Allocate device memory
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fRefImg, 
							   /*sizeof(uchar)**/refImg.rows*refImg.cols));
	checkCudaErrors(cudaMemcpy(m_cuHandle.m_d_fRefImg,
							   (void*)refImg.data,
							   /*sizeof(uchar)**/refImg.rows*refImg.cols,
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fTarImg, 
							   /*sizeof(uchar)**/refImg.rows*refImg.cols));

	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fU, 
							   sizeof(real_t)*iPOINum));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fV, 
							   sizeof(real_t)*iPOINum));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fZNCC, 
							   sizeof(real_t)*iPOINum));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_dev_FreqDom1, 
							   sizeof(cudafftComplex)*iPOINum*iFFTFreqSize));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_dev_FreqDom2, 
							   sizeof(cudafftComplex)*iPOINum*iFFTFreqSize));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_dev_FreqDomfg,
							   sizeof(cudafftComplex)*iPOINum*iFFTFreqSize));

	//!- Initialize the CUFFT plan
	int dim[2] = { iFFTSubW, iFFTSubH };
	int idim[2] = { iFFTSubW, iFFTSubH };
	int odim[2] = { iFFTSubW, (iFFTSubH / 2 + 1) };

	cufftPlanMany(&(m_cuHandle.m_forwardPlanXY), 
  				  2, dim,
				  idim, 1, iFFTSize,
				  odim, 1, iFFTFreqSize,
				  CUFFT_R2C, iPOINum);

	cufftPlanMany(&(m_cuHandle.m_reversePlanXY), 
				  2, dim,
				  odim, 1, iFFTFreqSize,
				  idim, 1, iFFTSize,
				  CUFFT_C2R, iPOINum);

	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fSubset1, 
							   sizeof(real_t)*iPOINum*iFFTSize));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fSubset2, 
							   sizeof(real_t)*iPOINum*iFFTSize));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fSubsetC,
							   sizeof(real_t)*iPOINum*iFFTSize));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fMod1, 
							   sizeof(real_t)*iPOINum));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fMod2, 
							   sizeof(real_t)*iPOINum));

	cudaDeviceSynchronize();
}

void cuFFTCC2D::ComputeFFTCC(// Output
							 real_t**& fU,
							 real_t**& fV,
							 real_t**& fZNCC,
							 // Input
							 const cv::Mat& tarImg)
{
	//if(tarImg.cols != )
	checkCudaErrors(cudaMemcpy(m_cuHandle.m_d_fTarImg,
							   (void*)tarImg.data,
							   /*sizeof(uchar)**/tarImg.cols*tarImg.rows,
							   cudaMemcpyHostToDevice));

	auto iFFTSubW = m_iSubsetX * 2;
	auto iFFTSubH = m_iSubsetY * 2;
	auto iPOINum = GetNumPOIs();

	if(!m_isWholeImgUsed)
		cufft_prepare_kernel <<<iPOINum, BLOCK_SIZE_256 >>> (m_cuHandle.m_d_iPOIXY,
															 m_cuHandle.m_d_fRefImg,
														 	 m_cuHandle.m_d_fTarImg,
															 iFFTSubH, iFFTSubW,
															 m_iSubsetX, m_iSubsetY,
															 m_iROIHeight, m_iROIWidth,
															 m_cuHandle.m_d_fSubset1,
															 m_cuHandle.m_d_fSubset2,
															 m_cuHandle.m_d_fMod1,
															 m_cuHandle.m_d_fMod2);
	else
		cufft_prepare_kernel <<<iPOINum, BLOCK_SIZE_256 >>> (m_cuHandle.m_d_iPOIXY,
															 m_cuHandle.m_d_fRefImg,
														 	 m_cuHandle.m_d_fTarImg,
															 iFFTSubH,     iFFTSubW,
															 m_iSubsetX,   m_iSubsetY,
															 m_iImgHeight, m_iImgWidth,
															 m_cuHandle.m_d_fSubset1,
															 m_cuHandle.m_d_fSubset2,
															 m_cuHandle.m_d_fMod1,
															 m_cuHandle.m_d_fMod2);
	getLastCudaError("Error in calling cufft_prepare_kernel");

	cufftExecR2C(m_cuHandle.m_forwardPlanXY, 
				 m_cuHandle.m_d_fSubset1, 
				 m_cuHandle.m_dev_FreqDom1);
	cufftExecR2C(m_cuHandle.m_forwardPlanXY, 
				 m_cuHandle.m_d_fSubset2,
				 m_cuHandle.m_dev_FreqDom2);

	complexMulandScale_kernel <<<iPOINum, BLOCK_SIZE_256 >>> (m_cuHandle.m_dev_FreqDom1,
															  m_cuHandle.m_dev_FreqDom2,
															  iFFTSubH, iFFTSubW,
															  m_cuHandle.m_d_fMod1,
															  m_cuHandle.m_d_fMod2,
															  m_cuHandle.m_dev_FreqDomfg);

	cufftExecC2R(m_cuHandle.m_reversePlanXY, 
				 m_cuHandle.m_dev_FreqDomfg, 
				 m_cuHandle.m_d_fSubsetC);

	findMax <<<iPOINum, BLOCK_SIZE_256 >>> (m_cuHandle.m_d_fSubsetC,
											iFFTSubH, iFFTSubW,
											m_iSubsetX, m_iSubsetY,
											m_cuHandle.m_d_fU,
											m_cuHandle.m_d_fV,
											m_cuHandle.m_d_fZNCC);

	checkCudaErrors(cudaMemcpy(fU[0],
							   m_cuHandle.m_d_fU,
							   sizeof(real_t)*iPOINum,
							   cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(fV[0], 
					           m_cuHandle.m_d_fV,
							   sizeof(real_t)*iPOINum,
							   cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(fZNCC[0], 
							   m_cuHandle.m_d_fZNCC, 
							   sizeof(real_t)*iPOINum, 
							   cudaMemcpyDeviceToHost));
}

void cuFFTCC2D::DestroyFFTCC(real_t**& fU,
							 real_t**& fV,
							 real_t**& fZNCC)
{
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fRefImg));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fTarImg));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fMod1));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fMod2));

	checkCudaErrors(cudaFree(m_cuHandle.m_dev_FreqDom1));
	checkCudaErrors(cudaFree(m_cuHandle.m_dev_FreqDom2));
	checkCudaErrors(cudaFree(m_cuHandle.m_dev_FreqDomfg));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fSubset1));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fSubset2));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fSubsetC));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fZNCC));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fU));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fV));

	cufftDestroy(m_cuHandle.m_forwardPlanXY);
	cufftDestroy(m_cuHandle.m_reversePlanXY);

	hdestroyptr<real_t>(fU);
	hdestroyptr<real_t>(fV);
	hdestroyptr<real_t>(fZNCC);

	isDestroyed = true;
}


void cuFFTCC2D::cuInitializeFFTCC(// Output
								  real_t *& f_d_U,
								  real_t *& f_d_V,
								  real_t*& f_d_ZNCC,
								  // Input
								  const cv::Mat& refImg)
{
	isLowLevelApiCalled = true;

	//!- Precompute the POI postions, since this is invariant during the entire
	// computation.
	//!- Determine whether the whole image or the ROI is used
	if(!m_isWholeImgUsed)
		cuComputePOIPositions(m_cuHandle.m_d_iPOIXY,
						 	  m_iNumPOIX, m_iNumPOIY,
					 		  m_iMarginX, m_iMarginY,
							  m_iSubsetX, m_iSubsetY,
							  m_iGridSpaceX, m_iGridSpaceY);
	else
		cuComputePOIPositions(m_cuHandle.m_d_iPOIXY,
							  m_iStartX, m_iStartY,
							  m_iNumPOIX, m_iNumPOIY,
							  m_iMarginX, m_iMarginY,
							  m_iSubsetX, m_iSubsetY,
							  m_iGridSpaceX, m_iGridSpaceY);

	int_t iPOINum = GetNumPOIs();
	int_t iROISize = GetROISize();
	int_t iFFTSubW = m_iSubsetX * 2, iFFTSubH = m_iSubsetY * 2;
	int_t iFFTSize = iFFTSubW * iFFTSubH;
	int_t iFFTFreqSize = iFFTSubW * (iFFTSubH / 2 + 1);

	//!- Allocate device memory
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fRefImg, 
							   /*sizeof(uchar)**/refImg.rows*refImg.cols));
	checkCudaErrors(cudaMemcpy(m_cuHandle.m_d_fRefImg,
							   (void*)refImg.data,
							   /*sizeof(uchar)**/refImg.rows*refImg.cols,
							   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fTarImg, 
							   /*sizeof(uchar)**/refImg.rows*refImg.cols));

	//!- Use these three parameters instead of the ones in m_cuHandle
	checkCudaErrors(cudaMalloc((void**)&f_d_U, 
							   sizeof(real_t)*iPOINum));
	checkCudaErrors(cudaMalloc((void**)&f_d_V, 
							   sizeof(real_t)*iPOINum));
	checkCudaErrors(cudaMalloc((void**)&f_d_ZNCC, 
							   sizeof(real_t)*iPOINum));

	//!- Initialize the CUFFT plan
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_dev_FreqDom1, 
							   sizeof(cudafftComplex)*iPOINum*iFFTFreqSize));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_dev_FreqDom2, 
							   sizeof(cudafftComplex)*iPOINum*iFFTFreqSize));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_dev_FreqDomfg,
							   sizeof(cudafftComplex)*iPOINum*iFFTFreqSize));

	int dim[2]  = { iFFTSubW, iFFTSubH };
	int idim[2] = { iFFTSubW, iFFTSubH };
	int odim[2] = { iFFTSubW, (iFFTSubH / 2 + 1) };

	cufftPlanMany(&(m_cuHandle.m_forwardPlanXY), 
				  2, dim,
				  idim, 1, iFFTSize,
				  odim, 1, iFFTFreqSize,
				  CUFFT_R2C, iPOINum);

	cufftPlanMany(&(m_cuHandle.m_reversePlanXY), 
				  2, dim,
				  odim, 1, iFFTFreqSize,
				  idim, 1, iFFTSize,
				  CUFFT_C2R, iPOINum);

	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fSubset1, 
							   sizeof(real_t)*iPOINum*iFFTSize));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fSubset2, 
							   sizeof(real_t)*iPOINum*iFFTSize));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fSubsetC,
							   sizeof(real_t)*iPOINum*iFFTSize));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fMod1, 
							   sizeof(real_t)*iPOINum));
	checkCudaErrors(cudaMalloc((void**)&m_cuHandle.m_d_fMod2, 
							   sizeof(real_t)*iPOINum));
}

void cuFFTCC2D::cuComputeFFTCC(// Output
							   real_t *& f_d_U,
							   real_t *& f_d_V,
							   real_t*& f_d_ZNCC,
							   // Input
							   const cv::Mat& tarImg)
{
	checkCudaErrors(cudaMemcpy(m_cuHandle.m_d_fTarImg,
							   (void*)tarImg.data,
							   /*sizeof(uchar)**/tarImg.cols*tarImg.rows,
							   cudaMemcpyHostToDevice));

	auto iFFTSubW = m_iSubsetX * 2;
	auto iFFTSubH = m_iSubsetY * 2;
	auto iPOINum = GetNumPOIs();

	if(!m_isWholeImgUsed)
		cufft_prepare_kernel <<<iPOINum, BLOCK_SIZE_256 >>> (m_cuHandle.m_d_iPOIXY,
															 m_cuHandle.m_d_fRefImg,
													 		 m_cuHandle.m_d_fTarImg,
															 iFFTSubH,     iFFTSubW,
															 m_iSubsetX,   m_iSubsetY,
															 m_iROIHeight, m_iROIWidth,
															 m_cuHandle.m_d_fSubset1,
															 m_cuHandle.m_d_fSubset2,
															 m_cuHandle.m_d_fMod1,
															 m_cuHandle.m_d_fMod2);
	else
		cufft_prepare_kernel <<<iPOINum, BLOCK_SIZE_256 >>> (m_cuHandle.m_d_iPOIXY,
															 m_cuHandle.m_d_fRefImg,
													 		 m_cuHandle.m_d_fTarImg,
															 iFFTSubH,     iFFTSubW,
															 m_iSubsetX,   m_iSubsetY,
															 m_iImgHeight, m_iImgWidth,
															 m_cuHandle.m_d_fSubset1,
															 m_cuHandle.m_d_fSubset2,
															 m_cuHandle.m_d_fMod1,
															 m_cuHandle.m_d_fMod2);
	getLastCudaError("Error in calling cufft_prepare_kernel");

	cufftExecR2C(m_cuHandle.m_forwardPlanXY, 
				 m_cuHandle.m_d_fSubset1, 
				 m_cuHandle.m_dev_FreqDom1);

	cufftExecR2C(m_cuHandle.m_forwardPlanXY, 
				 m_cuHandle.m_d_fSubset2,
				 m_cuHandle.m_dev_FreqDom2);

	complexMulandScale_kernel <<<iPOINum, BLOCK_SIZE_256 >>> (m_cuHandle.m_dev_FreqDom1,
															  m_cuHandle.m_dev_FreqDom2,
															  iFFTSubH, iFFTSubW,
															  m_cuHandle.m_d_fMod1,
															  m_cuHandle.m_d_fMod2,
															  m_cuHandle.m_dev_FreqDomfg);
	getLastCudaError("Error in calling complexMulandScale_kernel");

	cufftExecC2R(m_cuHandle.m_reversePlanXY, 
				 m_cuHandle.m_dev_FreqDomfg, 
				 m_cuHandle.m_d_fSubsetC);

	//!- Use the three arguments instead of the ones in m_cuHandle member
	findMax <<<iPOINum, BLOCK_SIZE_256 >>> (m_cuHandle.m_d_fSubsetC,
											iFFTSubH, iFFTSubW,
											m_iSubsetX, m_iSubsetY,
											f_d_U,
											f_d_V,
											f_d_ZNCC);
	getLastCudaError("Error in calling findMax_Kernel");
}

void  cuFFTCC2D::cuDestroyFFTCC(real_t *& f_d_U,
								real_t *& f_d_V,
								real_t*& f_d_ZNCC)
{
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fRefImg));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fTarImg));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fMod1));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fMod2));

	checkCudaErrors(cudaFree(m_cuHandle.m_dev_FreqDom1));
	checkCudaErrors(cudaFree(m_cuHandle.m_dev_FreqDom2));
	checkCudaErrors(cudaFree(m_cuHandle.m_dev_FreqDomfg));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fSubset1));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fSubset2));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fSubsetC));
	/*checkCudaErrors(cudaFree(m_cuHandle.m_d_fZNCC));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fU));
	checkCudaErrors(cudaFree(m_cuHandle.m_d_fV));*/

	cufftDestroy(m_cuHandle.m_forwardPlanXY);
	cufftDestroy(m_cuHandle.m_reversePlanXY);

	checkCudaErrors(cudaFree(f_d_U));
	checkCudaErrors(cudaFree(f_d_V));
	checkCudaErrors(cudaFree(f_d_ZNCC));

	isDestroyed = true;
}

void cuFFTCC2D::ResetRefImg(const cv::Mat& refImg)
{
	checkCudaErrors(cudaMemcpy(m_cuHandle.m_d_fRefImg,
							   (void*)refImg.data,
							   /*sizeof(uchar)**/refImg.rows*refImg.cols,
							   cudaMemcpyHostToDevice));
}

} //!- namespace paDIC
} //!- namespace TW