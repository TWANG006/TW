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

	void cuICGN2D_Initialize(uchar1 *m_d_fRefImg,
							 uchar1 *m_d_fTarImg,
							 int_t  *m_d_iPOIXY,
							 real_t *m_d_fU,
							 real_t *m_d_fV);
	
private:
	void cuICGN2D_prepare();

private:
	ICGN2DInterpolationFLag m_Iflag;

};

}// namespace paDIC
}// namespace TW

#endif // !TW_PADIC_CUICGN2D_H
