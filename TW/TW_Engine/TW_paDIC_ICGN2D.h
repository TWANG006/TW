#ifndef TW_PADIC_ICGN2D_H
#define TW_PADIC_ICGN2D_H

#include <TW.h>
#include <opencv2\opencv.hpp>

namespace TW{
namespace paDIC{

/// \brief ICGN2D base class fo 2D paDIC method
/// This class implement the general CPU-based ICGN algorithm. The computation
/// unit is based on two entire images. 
///	This class can be used as the basic class for multi-core
/// processing when used in paDIC algorithm.
/// NOTE: Currently only 1st-order shape function is considered.
/// TODO: 2nd-order shape function
class TW_LIB_DLL_EXPORTS ICGN2D
{
public:
	ICGN2D(const cv::Mat& refImg, 
		   const cv::Mat& tarImg,
		   int iStartX, int iStartY,
		   int iROIWidth, int iROIHeight,
		   int iSubsetX, int iSubsetY,
		   int iNumberX, int iNumberY,
		   int iNumIterations,
		   real_t fDeltaP);
	virtual ~ICGN2D();

	/*virtual void ICGN2D_Compute(real_t *fU,
								real_t *fV,
								int    *iPOIPos) = 0;*/
	virtual void ICGN2D_Precomputation_Prepare() = 0;
	virtual void ICGN2D_Precomputation() = 0;
	virtual void ICGN2D_Precomputation_Finalize() = 0;

protected:

	// Inputs
	cv::Mat m_refImg;	// Undeformed image
	cv::Mat m_tarImg;	// Deformed image
	int m_iStartX;
	int m_iStartY;
	int m_iROIWidth;
	int m_iROIHeight;
	int m_iSubsetX;
	int m_iSubsetY;
	int m_iNumberX;
	int m_iNumberY;
	int m_iPOINumber;
	int m_iNumIterations;
	real_t m_fDeltaP;

	// Parameters for computation

};

} //!- namespace paDIC
} //!- namespace TW

#endif // !TW_PADIC_ICGN2D_H
