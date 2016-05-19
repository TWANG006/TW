#ifndef TW_paDIC_ICGN2D_CPU_H
#define TW_paDIC_ICGN2D_CPU_H

#include "opencv2\opencv.hpp"
#include "TW_paDIC_ICGN2D.h"


namespace TW{
namespace paDIC{

/// \brief ICGN2D implementation based on CPU computation
/// Launch Order: 
/// 1. ICGN2D_Precomputation_Prepare() // Allocate space for precomputation
/// 2. ICNG2D_Prepare()
/// 3. ICGN2D_Precomputation()
/// 4. ICNG2D_Compute()
/// 5. ICGN_Precomputation_Finalize()
/// 6. ICGN_Finalize()
/// 7. ICGN_Precomputation_Finalize()
class TW_LIB_DLL_EXPORTS ICGN2D_CPU : public ICGN2D
{
public:
	ICGN2D_CPU(/*(const cv::Mat& refImg,
			   const cv::Mat& tarImg,*/
			   //const cv::Mat& refImg,
			   int_t iImgWidth, int_t iImgHeight,
			   int_t iStartX, int_t iStartY,
			   int_t iROIWidth, int_t iROIHeight,
			   int_t iSubsetX, int_t iSubsetY,
			   int_t iNumberX, int_t iNumberY,
		       int_t iNumIterations,
		       real_t fDeltaP,
			   ICGN2DInterpolationFLag Iflag,
			   ICGN2DThreadFlag Tflag);

	ICGN2D_CPU(/*(const cv::Mat& refImg,
			   const cv::Mat& tarImg,*/
			   const cv::Mat& refImg,
			   int_t iImgWidth, int_t iImgHeight,
			   int_t iStartX, int_t iStartY,
			   int_t iROIWidth, int_t iROIHeight,
			   int_t iSubsetX, int_t iSubsetY,
			   int_t iNumberX, int_t iNumberY,
		       int_t iNumIterations,
		       real_t fDeltaP,
			   ICGN2DInterpolationFLag Iflag,
			   ICGN2DThreadFlag Tflag);
	~ICGN2D_CPU();

	virtual void ResetRefImg(const cv::Mat& refImg) override;


	void ICGN2D_Algorithm(real_t *fU,
						  real_t *fV,
						  int *iNumIterations,
						  const int *iPOIpos,
						  const cv::Mat& tarImg);


	///\brief Core ICGN2D algorithm. The calculation unit is subset.
	///
	///\param fU Initial guees for the displacement in U(x) direction
	///\param fV Initial guees for the displacement in V(Y) direction
	///\param iPOIx POI x-position 
	///\param iPOIy POI y-position
	///\param id the id of the current POI being processed (For multi-threaded computation)
	ICGN2DFlag ICGN2D_Compute(real_t &fU,
							  real_t &fV,
							  int &iNumIterations,
							  const int iPOIx,
							  const int iPOIy,
							  const int id);

	

private:
	void ICGN2D_Precomputation_Prepare();
	void ICGN2D_Precomputation();
	void ICGN2D_Precomputation_Finalize();
	///\brief Allocate required memory for the ICNG2D application
	void ICGN2D_Prepare();
	void ICGN2D_Finalize();
	

private:
	ICGN2DInterpolationFLag m_Iflag;
	ICGN2DThreadFlag m_Tflag;
	real_t **m_fRx;						// x-derivative of reference image ROI_H * ROI_W
	real_t **m_fRy;						// y-derivative of reference image ROI_H * ROI_W
	real_t **m_fTx;						// Only for Bicubic Inerpolation 
	real_t **m_fTy;						// Only for Bicubic Inerpolation
	real_t **m_fTxy;					// Only for Bicubic Inerpolation
	real_t ****m_fInterpolation;	    // LUT for Interpolation: ROI_H * ROI_W * 4 * 4
	real_t ***m_fSubsetR;				// POI_N * Subset_H * Subset_W
	real_t ***m_fSubsetT;				// POI_N * Subset_H * Subset_W
	real_t ****m_fRDescent;				// POI_N * Subset_H * Subset_W * 6;
};

}
}

#endif // !TW_paDIC_ICGN2D_CPU_H
