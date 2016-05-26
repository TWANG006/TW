#include "TW_paDIC_FFTCC2D_CPU.h"
#include "TW_utils.h"
#include "TW_MemManager.h"

namespace TW{
namespace paDIC{

Fftcc2D_CPU::Fftcc2D_CPU(const int_t iImgWidth, const int_t iImgHeight,
					 	 const int_t iStartX, const int_t iStartY,
						 const int_t iROIWidth, const int_t iROIHeight,
						 const int_t iSubsetX, const int_t iSubsetY,
						 const int_t iGridSpaceX, const int_t iGridSpaceY,
						 const int_t iMarginX, const int_t iMarginY,
						 paDICThreadFlag tFlag)
						 : Fftcc2D(iImgWidth, iImgHeight,
							 	   iStartX, iStartY,
								   iROIWidth, iROIHeight,
								   iSubsetX, iSubsetY,
								   iGridSpaceX, iGridSpaceY,
								   iMarginX, iMarginY)
						 , m_tFlag(tFlag)
						 , m_isDestroyed(false)
{
	m_iFFTSubsetH = 2 * iSubsetY;
	m_iFFTSubsetW = 2 * iSubsetX;

	if (!recomputeNumPOI())
		throw std::logic_error("Number of POIs is below 0!");
	else
	{
		// Make the FFTCC plans
		m_iPOINum = GetNumPOIs();

		hcreateptr(m_fSubset1, m_iPOINum * m_iFFTSubsetW * m_iFFTSubsetH);
		hcreateptr(m_fSubset2, m_iPOINum * m_iFFTSubsetW * m_iFFTSubsetH);
		hcreateptr(m_fSubsetC, m_iPOINum * m_iFFTSubsetW * m_iFFTSubsetH);

		m_freqDom1 = (fftw3Complex*)fftw_malloc(sizeof(fftw3Complex)*m_iPOINum*m_iFFTSubsetW*(m_iFFTSubsetH / 2 + 1));
		m_freqDom2 = (fftw3Complex*)fftw_malloc(sizeof(fftw3Complex)*m_iPOINum*m_iFFTSubsetW*(m_iFFTSubsetH / 2 + 1));
		m_freqDomC = (fftw3Complex*)fftw_malloc(sizeof(fftw3Complex)*m_iPOINum*m_iFFTSubsetW*(m_iFFTSubsetH / 2 + 1));

		m_fftwPlan1 = new fftw3Plan[m_iPOINum];
		m_fftwPlan2 = new fftw3Plan[m_iPOINum];
		m_rfftwPlan = new fftw3Plan[m_iPOINum];

		for (int i = 0; i < m_iPOINum; i++)
		{
#ifdef TW_USE_DOUBLE
			m_fftwPlan1[i] = fftw_plan_dft_r2c_2d(m_iFFTSubsetW, m_iFFTSubsetH, 
												  &m_fSubset1[i*m_iFFTSubsetH*m_iFFTSubsetW], 
												  &m_freqDom1[i*m_iFFTSubsetW*(m_iFFTSubsetH / 2 + 1)],
												  FFTW_ESTIMATE);
			m_fftwPlan2[i] = fftw_plan_dft_r2c_2d(m_iFFTSubsetW, m_iFFTSubsetH, 
												  &m_fSubset2[i*m_iFFTSubsetH*m_iFFTSubsetW], 
												  &m_freqDom2[i*m_iFFTSubsetW*(m_iFFTSubsetH / 2 + 1)],
												  FFTW_ESTIMATE);
			m_rfftwPlan[i] = fftw_plan_dft_c2r_2d(m_iFFTSubsetW, m_iFFTSubsetH, 
												  &m_freqDomC[i*m_iFFTSubsetW*(m_iFFTSubsetH / 2 + 1)], 
												  &m_fSubsetC[i*m_iFFTSubsetH*m_iFFTSubsetW],
												  FFTW_ESTIMATE);
#else
			m_fftwPlan1[i] = fftwf_plan_dft_r2c_2d(m_iFFTSubsetW, m_iFFTSubsetH, 
												   &m_fSubset1[i*m_iFFTSubsetH*m_iFFTSubsetW], 
												   &m_freqDom1[i*m_iFFTSubsetW*(m_iFFTSubsetH / 2 + 1)],
												   FFTW_ESTIMATE);
			m_fftwPlan2[i] = fftwf_plan_dft_r2c_2d(m_iFFTSubsetW, m_iFFTSubsetH, 
												   &m_fSubset2[i*m_iFFTSubsetH*m_iFFTSubsetW], 
												   &m_freqDom2[i*m_iFFTSubsetW*(m_iFFTSubsetH / 2 + 1)],
												   FFTW_ESTIMATE);
			m_rfftwPlan[i] = fftwf_plan_dft_c2r_2d(m_iFFTSubsetW, m_iFFTSubsetH, 
												   &m_freqDomC[i*m_iFFTSubsetW*(m_iFFTSubsetH / 2 + 1)], 
												   &m_fSubsetC[i*m_iFFTSubsetH*m_iFFTSubsetW],
												   FFTW_ESTIMATE);
#endif // TW_USE_DOUBLE

		}
	}

}
Fftcc2D_CPU::~Fftcc2D_CPU()
{
	if(!m_isDestroyed)
	{
		hdestroyptr(m_fSubset1);
		hdestroyptr(m_fSubset2);
		hdestroyptr(m_fSubsetC);

		for (int i = 0; i < m_iPOINum; i++)
		{
#ifdef TW_USE_DOUBLE
			fftw_destroy_plan(m_fftwPlan1[i]);
			fftw_destroy_plan(m_fftwPlan2[i]);
			fftw_destroy_plan(m_rfftwPlan[i]);
#else
			fftwf_destroy_plan(m_fftwPlan1[i]);
			fftwf_destroy_plan(m_fftwPlan2[i]);
			fftwf_destroy_plan(m_rfftwPlan[i]);
#endif // TW_USE_DOUBLE
		}

		deleteObject(m_fftwPlan1);
		deleteObject(m_fftwPlan2);
		deleteObject(m_rfftwPlan);

#ifdef TW_USE_DOUBLE
		fftw_free(m_freqDom1);
		fftw_free(m_freqDom2);
		fftw_free(m_freqDomC);
#else
		fftwf_free(m_freqDom1);
		fftwf_free(m_freqDom2);
		fftwf_free(m_freqDomC);
#endif // TW_USE_DOUBLE
	}
}

void Fftcc2D_CPU::ResetRefImg(const cv::Mat& refImg)
{
	m_refImg = refImg;
}

void Fftcc2D_CPU::SetTarImg(const cv::Mat& tarImg)
{
	m_tarImg = tarImg;
}

void Fftcc2D_CPU::InitializeFFTCC(// Inputs
								  const cv::Mat& refImg,
								  // Outputs
								  int_t ***& iPOIXY,
								  real_t**& fU,
								  real_t**& fV,
								  real_t**& fZNCC)
{
	// Assign the refImg
	m_refImg = refImg;

	// Allocate memory for fU, fV & fZNCC
	hcreateptr<real_t>(fU, m_iNumPOIY, m_iNumPOIX);
	hcreateptr<real_t>(fV, m_iNumPOIY, m_iNumPOIX);
	hcreateptr<real_t>(fZNCC, m_iNumPOIY, m_iNumPOIX);

	// Compute the POI positions
	switch (m_tFlag)
	{
	case TW::paDIC::paDICThreadFlag::Single:
	{
		ComputePOIPositions_s(iPOIXY, 
							  m_iStartX, m_iStartY,
							  m_iNumPOIX, m_iNumPOIY,
							  m_iMarginX, m_iMarginY,
							  m_iSubsetX, m_iSubsetY,
							  m_iGridSpaceX, m_iGridSpaceY);

		break;
	}

	case TW::paDIC::paDICThreadFlag::Multicore:
	{
		ComputePOIPositions_m(iPOIXY,
							  m_iStartX, m_iStartY,
							  m_iNumPOIX, m_iNumPOIY,
							  m_iMarginX, m_iMarginY,
							  m_iSubsetX, m_iSubsetY,
							  m_iGridSpaceX, m_iGridSpaceY);

		break;
	}
	default:
		break;
	} 
}

void Fftcc2D_CPU::Algorithm_FFTCC(// Inputs
						 const cv::Mat& tarImg,
						 int_t*** const& iPOIXY,
						 // Outputs
						 real_t**& fU,
						 real_t**& fV,
						 real_t**& fZNCC)
{
	m_tarImg = tarImg;

	switch (m_tFlag)
	{
	case TW::paDIC::paDICThreadFlag::Single:
	{	
		for (int i = 0; i < m_iPOINum; i++)
		{
			int x = i % m_iNumPOIX;
			int y = i / m_iNumPOIX;
			ComputeFFTCC(iPOIXY[y][x],
						 i,
						 fU[y][x],
						 fV[y][x],
						 fZNCC[y][x]);
		}

		break;
	}
	case TW::paDIC::paDICThreadFlag::Multicore:
	{
#pragma omp parallel for
		for (int i = 0; i < m_iPOINum; i++)
		{
			int x = i % m_iNumPOIX;
			int y = i / m_iNumPOIX;
			ComputeFFTCC(iPOIXY[y][x],
						 i,
						 fU[y][x],
						 fV[y][x],
						 fZNCC[y][x]);
		}
		break;
	}
	default:
		break;
	}
}


void Fftcc2D_CPU::ComputeFFTCC(// Inputs
							   const int_t *iPOIXY,
							   const int id,
					           // Outputs
							   real_t &fU,
							   real_t &fV,
							   real_t &fZNCC)
{
	int iFFTSize = m_iFFTSubsetW * m_iFFTSubsetH;
	int iFFTFreqSize = m_iFFTSubsetW * (m_iFFTSubsetH / 2 + 1);

	real_t fSubAveR, fSubAveT, fSubNorR, fSubNorT;

	fSubAveR = 0;	// R_m
	fSubAveT = 0;	// T_m

	// Fill the intensity values into the subsets in refImg & tarImg
	for (int i = 0; i < m_iFFTSubsetH; i++)
	{
		for (int j = 0; j < m_iFFTSubsetW; j++)
		{
			real_t tempSubset;
			tempSubset = m_fSubset1[id*iFFTSize + i*m_iFFTSubsetW + j] = 
				static_cast<real_t>(m_refImg.at<uchar>(iPOIXY[0] - m_iSubsetY + i, iPOIXY[1] - m_iSubsetX + j));
			fSubAveR += tempSubset;

			tempSubset = m_fSubset2[id*iFFTSize + i*m_iFFTSubsetW + j] = 
				static_cast<real_t>(m_tarImg.at<uchar>(iPOIXY[0] - m_iSubsetY + i, iPOIXY[1] - m_iSubsetX + j));
			fSubAveT += tempSubset;
		}
	}
	fSubAveR = fSubAveR / real_t(iFFTSize);
	fSubAveT = fSubAveT / real_t(iFFTSize);

	// Compute the R_i - R_m & T_i - T_m
	fSubNorR = 0;	// sqrt(sigma(R_i - R_m)^2)
	fSubNorT = 0;	// sqrt(sigma(T_i - T_m)^2)
	for (int i = 0; i < m_iFFTSubsetH; i++)
	{
		for (int j = 0; j < m_iFFTSubsetW; j++)
		{
			real_t tempSubset;
			tempSubset = m_fSubset1[id*iFFTSize + i*m_iFFTSubsetW + j] -= fSubAveR;
			fSubNorR += tempSubset * tempSubset;

			tempSubset = m_fSubset2[id*iFFTSize + i*m_iFFTSubsetW + j] -= fSubAveT;
			fSubNorT += tempSubset * tempSubset;
		}
	}
	// Terminate the processing if subsets are full of zero intencities
	if (fSubNorR < 0.0000001 || fSubNorT < 0.0000001)
	{
		return;
	}

	// FFT-CC Algorithm using FFTW3
#ifdef TW_USE_DOUBLE
	fftw_execute(m_fftwPlan1[id]);
	fftw_execute(m_fftwPlan2[id]);
#else
	fftwf_execute(m_fftwPlan1[id]);
	fftwf_execute(m_fftwPlan2[id]);
#endif // TW_USE_DOUBLE
	for (int p = 0; p < m_iFFTSubsetW * (m_iFFTSubsetH / 2 + 1); p++)
	{
		m_freqDomC[id*iFFTFreqSize + p][0] =
			(m_freqDom1[id*iFFTFreqSize + p][0] * m_freqDom2[id*iFFTFreqSize + p][0]) +
			(m_freqDom1[id*iFFTFreqSize + p][1] * m_freqDom2[id*iFFTFreqSize + p][1]);
		m_freqDomC[id*iFFTFreqSize + p][1] =
			(m_freqDom1[id*iFFTFreqSize + p][0] * m_freqDom2[id*iFFTFreqSize + p][1]) -
			(m_freqDom1[id*iFFTFreqSize + p][1] * m_freqDom2[id*iFFTFreqSize + p][0]);
	}
#ifdef TW_USE_DOUBLE
	fftw_execute(m_rfftwPlan[id]);
#else
	fftwf_execute(m_rfftwPlan[id]);
#endif // TW_USE_DOUBLE

	fZNCC = -2;	// Maximum ZNCC value
	int_t iCorrPeak = 0;	// Index of the Maximum ZNCC value

	// Seach for maximum C
	for (int k = 0; k < iFFTSize; k++)
	{
		if (fZNCC < m_fSubsetC[id*iFFTSize + k])
		{
			fZNCC = m_fSubsetC[id*iFFTSize + k];
			iCorrPeak = k;
		}
	}

	fZNCC /= sqrt(fSubNorR * fSubNorT)*real_t(iFFTSize);	// Normalization parameter

	// Calculate the location of maximum C
	int iU = iCorrPeak % m_iFFTSubsetW;	// x
	int iV = iCorrPeak / m_iFFTSubsetW;	// y

	if(iU > m_iSubsetX)
		iU -= m_iFFTSubsetW;
	if(iV > m_iSubsetY)
		iV -= m_iFFTSubsetH;

	fU = (real_t)iU;
	fV = (real_t)iV;
}

void Fftcc2D_CPU::FinalizeFFTCC(int_t ***& iPOIXY,
								real_t **& fU,
								real_t **& fV,
								real_t **& fZNCC)
{
	hdestroyptr(iPOIXY);
	hdestroyptr(fU);
	hdestroyptr(fV);
	hdestroyptr(fZNCC);

	hdestroyptr(m_fSubset1);
	hdestroyptr(m_fSubset2);
	hdestroyptr(m_fSubsetC);

	for (int i = 0; i < m_iPOINum; i++)
	{
#ifdef TW_USE_DOUBLE
		fftw_destroy_plan(m_fftwPlan1[i]);
		fftw_destroy_plan(m_fftwPlan2[i]);
		fftw_destroy_plan(m_rfftwPlan[i]);
#else
		fftwf_destroy_plan(m_fftwPlan1[i]);
		fftwf_destroy_plan(m_fftwPlan2[i]);
		fftwf_destroy_plan(m_rfftwPlan[i]);
#endif // TW_USE_DOUBLE
	}

	deleteObject(m_fftwPlan1);
	deleteObject(m_fftwPlan2);
	deleteObject(m_rfftwPlan);

#ifdef TW_USE_DOUBLE
	fftw_free(m_freqDom1);
	fftw_free(m_freqDom2);
	fftw_free(m_freqDomC);
#else
	fftwf_free(m_freqDom1);
	fftwf_free(m_freqDom2);
	fftwf_free(m_freqDomC);
#endif // TW_USE_DOUBLE

	m_isDestroyed = true;

}

} // namespace paDIC
} // namespace TW
