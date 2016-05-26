#ifndef TW_PADIC_FFTCC2D_CPU_H
#define TW_PADIC_FFTCC2D_CPU_H

#include "TW.h"
#include "TW_paDIC.h"
#include "TW_paDIC_FFTCC2D.h"

namespace TW{
namespace paDIC{

class TW_LIB_DLL_EXPORTS Fftcc2D_CPU : public Fftcc2D
{
public:
	Fftcc2D_CPU(const int_t iImgWidth, const int_t iImgHeight,
				const int_t iStartX, const int_t iStartY,
				const int_t iROIWidth, const int_t iROIHeight,
				const int_t iSubsetX, const int_t iSubsetY,
				const int_t iGridSpaceX, const int_t iGridSpaceY,
				const int_t iMarginX, const int_t iMarginY,
				paDICThreadFlag tFlag);
	~Fftcc2D_CPU();

	virtual void ResetRefImg(const cv::Mat& refImg) override;
	virtual void SetTarImg(const cv::Mat& refImg) override;

	void InitializeFFTCC(// Inputs
						 const cv::Mat& refImg,
						 // Outputs
						 int_t ***& iPOIXY,
						 real_t**& fU,
						 real_t**& fV,
						 real_t**& fZNCC);

	void Algorithm_FFTCC(// Inputs
						 const cv::Mat& tarImg,
						 int_t*** const& iPOIXY,
						 // Outputs
						 real_t**& fU,
						 real_t**& fV,
						 real_t**& fZNCC);

	void ComputeFFTCC(// Inputs
					  const int_t *iPOIXY,
					  const int id,
					  // Outputs
					  real_t &fU,
					  real_t &fV,
					  real_t &fZNCC);

	void FinalizeFFTCC(int_t ***& iPOIXY,
					   real_t **& fU,
					   real_t **& fV,
					   real_t **& fZNCC);

private:
	paDICThreadFlag m_tFlag;
	bool m_isDestroyed;

	cv::Mat m_refImg;	// Ref Img
	cv::Mat m_tarImg;	// Tar Img

	int m_iFFTSubsetW;
	int m_iFFTSubsetH;
	int m_iPOINum;

	real_t *m_fSubset1;
	fftw3Complex *m_freqDom1;
	real_t *m_fSubset2;
	fftw3Complex *m_freqDom2;
	real_t *m_fSubsetC;
	fftw3Complex *m_freqDomC;

	fftw3Plan *m_fftwPlan1;
	fftw3Plan *m_fftwPlan2;
	fftw3Plan *m_rfftwPlan;
};

} // namespace paDIC
} // namespace TW


#endif // !TW_PADIC_FFTCC2D_CPU_H
