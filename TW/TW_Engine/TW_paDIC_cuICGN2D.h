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

	void cuCompute(uchar1 *d_fTarImg,
				   int_t  *d_iPOIXY,
				   real_t *d_fU,
				   real_t *d_fV);

	void cuInitialize(uchar1 *d_fRefImg);
	void Initialize(cv::Mat& refImg);
	
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
