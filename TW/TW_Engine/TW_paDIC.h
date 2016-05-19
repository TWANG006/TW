#ifndef TW_PADIC_H
#define TW_PADIC_H

#include "opencv2\opencv.hpp"
#include "TW.h"

namespace TW{
namespace paDIC{

/// \struct GPUHandle_FFTCC
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

	GPUHandle_FFTCC();
};

/// \struct GPUHandle
/// \brief This handle is for the computation of FFTCC of paDIC on GPU using CUDA
/// This struct holds all the needed parameters of the parallel FFTCC algorithm.
struct GPUHandle_ICGN
{
	uchar1 *m_d_fRefImg;			// Reference image
	uchar1 *m_d_fTarImg;			// Target image

	int_t  *m_d_iPOIXY;				// POI positions on device
	real_t *m_d_fU;					// Displacement in x-direction
	real_t *m_d_fV;					// Displacement in y-direction
	/*-----The above paramters may be aquired from FFT-CC algorithm-----*/
	
	// ICGN calculation parameters
	real_t *m_d_fRx;
	real_t *m_d_fRy;
	real_t *m_d_fTx;
	real_t *m_d_fTy;
	real_t *m_d_fTxy;
	float4* m_d_f4InterpolationLUT;

	int *m_d_iIterationNums;

	real_t *m_d_fSubsetR;
	real_t *m_d_fSubsetT;
	real_t *m_d_fSubsetAveR;
	real_t *m_d_fSubsetAveT;
	real_t *m_d_Hessian;
	real_t *m_d_RDescent;
	
	GPUHandle_ICGN();
};

//extern TW_LIB_DLL_EXPORTS GPUHandle_FFTCC g_cuHandle;

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
