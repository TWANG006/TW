#ifndef TW_PADIC_H
#define TW_PADIC_H

#include "opencv2\opencv.hpp"
#include "TW.h"

namespace TW{
namespace paDIC{

/// \struct GPUHandle
/// \brief This handle is for the computation of FFTCC of paDIC on GPU using CUDA
/// This struct holds all the needed parameters of the parallel FFTCC algorithm.
struct GPUHandle_FFTCC
{
	uchar1 *m_d_fRefImg;			// Reference image
	uchar1 *m_d_fTarImg;			// Target image

	int_t *m_d_iPOIXY;				// POI positions on device
	real_t *m_d_fU;					// Displacement in x-direction
	real_t *m_d_fV;					// Displacement in y-direction
	real_t *m_d_fZNCC;				// ZNCC coefficients of each subset

	// Fourier Transform needed parameters
	real_t *m_d_fSubset1;
	real_t *m_d_fSubset2;
	real_t *m_d_fSubsetC;
	real_t *m_d_fMod1;
	real_t *m_d_fMod2;
	cufftHandle m_forwardPlanXY;
	cufftHandle m_reversePlanXY;
	cudafftComplex *m_dev_FreqDom1;
	cudafftComplex *m_dev_FreqDom2;
	cudafftComplex *m_dev_FreqDomfg;

	GPUHandle_FFTCC()
		:m_d_fRefImg(nullptr), m_d_fTarImg(nullptr), m_d_iPOIXY(nullptr),
		 m_d_fV(nullptr), m_d_fU(nullptr), m_d_fZNCC(nullptr),
		 m_d_fSubset1(nullptr), m_d_fSubset2(nullptr), m_d_fSubsetC(nullptr),
		 m_d_fMod1(nullptr), m_d_fMod2(nullptr),
		 m_dev_FreqDom1(nullptr), m_dev_FreqDom2(nullptr), m_dev_FreqDomfg(nullptr)
	{}
};

extern TW_LIB_DLL_EXPORTS GPUHandle_FFTCC g_cuHandle;

enum class ICGN2DFlag
{
	Success,
	DarkSubset,
	SingularHessian,
	SingularWarp,
	OutofROI
};

enum class ICGN2DThreadFlag
{
	Single,
	Multicore
};

enum class ICGN2DInterpolationFLag
{
	Bicubic,
	BicubicSpline
};

}
}

#endif // !TW_PADIC_H
