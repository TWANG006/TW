#ifndef TW_PADIC_ICGN2D_H
#define TW_PADIC_ICGN2D_H

#include "TW.h"
#include "TW_paDIC.h"
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

	ICGN2D(/*const cv::Mat& refImg, 
		   const cv::Mat& tarImg,*/
		   //const cv::Mat& refImg,
		   int_t iImgWidth, int_t iImgHeight,
		   int_t iStartX, int_t iStartY,
		   int_t iROIWidth, int_t iROIHeight,
		   int_t iSubsetX, int_t iSubsetY,
		   int_t iNumberX, int_t iNumberY,
		   int_t iNumIterations,
		   real_t fDeltaP);

	ICGN2D(/*const cv::Mat& refImg, 
		   const cv::Mat& tarImg,*/
		   const cv::Mat& refImg,
		   int_t iImgWidth, int_t iImgHeight,
		   int_t iStartX, int_t iStartY,
		   int_t iROIWidth, int_t iROIHeight,
		   int_t iSubsetX, int_t iSubsetY,
		   int_t iNumberX, int_t iNumberY,
		   int_t iNumIterations,
		   real_t fDeltaP);

	virtual ~ICGN2D() = 0;	// To make this class an abstract base-class

	virtual void setROI(const int_t& iStartX,   const int_t& iStartY,
						const int_t& iROIWidth, const int_t& iROIHeight);

	virtual void ResetRefImg(const cv::Mat& refImg) = 0;


	/*virtual void ICGN2D_Compute(real_t *fU,
								real_t *fV,
								int    *iPOIPos) = 0;*/
	/*virtual void ICGN2D_Precomputation_Prepare() = 0;
	virtual void ICGN2D_Precomputation() = 0;
	virtual void ICGN2D_Precomputation_Finalize() = 0;*/

protected:

	// Inputs
	cv::Mat m_refImg;		// Undeformed image
	cv::Mat m_tarImg;		// Deformed image

	bool m_isRefImgUpdated;	// Flag to monitor whether the reference image is chanaged
	int_t m_iImgWidth;
	int_t m_iImgHeight;
	int_t m_iStartX;
	int_t m_iStartY;
	int_t m_iROIWidth;
	int_t m_iROIHeight;
	int_t m_iSubsetX;
	int_t m_iSubsetY;
	int_t m_iSubsetH;
	int_t m_iSubsetW;
	int_t m_iSubsetSize;
	int_t m_iNumberX;
	int_t m_iNumberY;
	int_t m_iPOINumber;
	int_t m_iNumIterations;	// Max number of allowed iterations
	real_t m_fDeltaP;		// Threshold of the convergence

	// Parameters for computation

};

} //!- namespace paDIC
} //!- namespace TW

#endif // !TW_PADIC_ICGN2D_H
