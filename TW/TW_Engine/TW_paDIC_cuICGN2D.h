#ifndef TW_PADIC_CUICGN2D_H
#define TW_PADIC_CUICGN2D_H

#include "TW_paDIC_ICGN2D.h"

namespace TW{
namespace paDIC{
	
class TW_LIB_DLL_EXPORTS cuICGN2D : public ICGN2D
{
public:
	GPUHandle_ICGN g_cuHandleICGN;

	cuICGN2D(//const cv::Mat& refImg,
			 int_t iImgWidth, int_t iImgHeight,
			 int_t iStartX, int_t iStartY,
			 int_t iROIWidth, int_t iROIHeight,
			 int_t iSubsetX, int_t iSubsetY,
			 int_t iNumberX, int_t iNumberY,
			 int_t iNumIterations,
			 real_t fDeltaP,
			 ICGN2DInterpolationFLag Iflag);

	~cuICGN2D();
	
	/// \brief Initialize ICGN with d_fRefImg.
	/// Note: The d_fRefImg is on GPU and it has already been filled with data. 
	/// The d_fRefImg's pointer is passed to g_cuHandleICGN. 
	/// 
	/// \param d_fRefImg The refImg on GPU side.
	void cuInitialize(uchar1 *d_fRefImg);

	/// \brief 
	void cuCompute(// Inputs
				   uchar1 *d_fTarImg,
				   int_t  *d_iPOIXY,
				   // Inputs & Outputs
				   real_t *d_fU,
				   real_t *d_fV);

	/// \brief Free memory allocated on GPU
	void cuFinalize();

	void Initialize(cv::Mat& refImg);
	void ResetRefImg(const cv::Mat& refImg) override;
	
private:
	/// \brief Allocate memory on GPU for the computation
	void prepare();

private:
	ICGN2DInterpolationFLag m_Iflag;
	bool m_isRefImgUpdated;

};

}// namespace paDIC
}// namespace TW

#endif // !TW_PADIC_CUICGN2D_H
