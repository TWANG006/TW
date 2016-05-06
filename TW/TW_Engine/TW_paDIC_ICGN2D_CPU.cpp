#include "TW_paDIC_ICGN2D_CPU.h"
#include "TW_MemManager.h"
#include "TW_utils.h"

namespace TW{
namespace paDIC{

ICGN2D_CPU::ICGN2D_CPU(const cv::Mat& refImg,
					   const cv::Mat& tarImg,
					   int iStartX, int iStartY,
					   int iROIWidth, int iROIHeight,
					   int iSubsetX, int iSubsetY,
					   int iNumberX, int iNumberY,
					   int iNumIterations,
					   real_t fDeltaP)
	: ICGN2D(refImg, 
		 	 tarImg,
		 	 iStartX, iStartY,
			 iROIWidth, iROIHeight,
			 iSubsetX, iSubsetY,
			 iNumberX, iNumberY,
			 iNumIterations,
			 fDeltaP)
{

}

ICGN2D_CPU::~ICGN2D_CPU()
{
}

void ICGN2D_CPU::ICGN2D_Precomputation_Prepare()
{
	hcreateptr<real_t>(m_fRx, m_iROIHeight, m_iROIWidth);
	hcreateptr<real_t>(m_fRy, m_iROIHeight, m_iROIWidth);
	hcreateptr<real_t>(m_fBsplineInterpolation, m_iROIHeight, m_iROIWidth, 4, 4);
}

void ICGN2D_CPU::ICGN2D_Precomputation() 
{
	// Compute gradients of m_refImg
	Gradient_s(m_refImg, 
			   m_iStartX, m_iStartY, 
			   m_iROIWidth, m_iROIHeight,
			   m_refImg.cols, m_refImg.rows,
			   TW::Quadratic,
			   m_fRx,
			   m_fRy);

	// Compute the LUT for bicubic B-Spline interpolation
	BicubicSplineCoefficients_s(m_tarImg,
								m_iStartX,
								m_iStartY,
								m_iROIWidth,
								m_iROIHeight,
								m_tarImg.cols,
								m_tarImg.rows,
								m_fBsplineInterpolation);
}

void ICGN2D_CPU::ICGN2D_Precomputation_Finalize()
{
	hdestroyptr(m_fRx);
	hdestroyptr(m_fRy);
	hdestroyptr(m_fBsplineInterpolation);
}

void ICGN2D_CPU::ICGN2D_Prepare()
{
	hcreateptr(m_fSubsetR, m_iPOINumber, m_iSubsetH, m_iSubsetW);
	hcreateptr(m_fSubsetT, m_iPOINumber, m_iSubsetH, m_iSubsetW);
	hcreateptr(m_fRDescent,m_iPOINumber, m_iSubsetH, m_iSubsetW, 6);
}

void ICGN2D_CPU::ICGN2D_Compute(real_t &fU,
								real_t &fV,
								const int iPOIx,
								const int iPOIy,
								const int id)
{

}

void ICGN2D_CPU::ICGN2D_Finalize()
{
	hdestroyptr(m_fSubsetR);
	hdestroyptr(m_fSubsetT);
	hdestroyptr(m_fRDescent);
}

} //!- namespace paDIC
} //!- namespace TW