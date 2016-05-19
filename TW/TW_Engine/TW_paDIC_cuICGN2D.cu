#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"

#include "TW_paDIC_cuICGN2D.h"

namespace TW{
namespace paDIC{

cuICGN2D::cuICGN2D(//const cv::Mat& refImg,
				   int_t iImgWidth, int_t iImgHeight,
				   int_t iStartX, int_t iStartY,
				   int_t iROIWidth, int_t iROIHeight,
				   int_t iSubsetX, int_t iSubsetY,
				   int_t iNumberX, int_t iNumberY,
				   int_t iNumIterations,
				   real_t fDeltaP,
				   ICGN2DInterpolationFLag Iflag)
			 : ICGN2D(//refImg,
					 iImgWidth, iImgHeight,
					 iStartX, iStartY,
					 iROIWidth, iROIHeight,
					 iSubsetX, iSubsetY,
					 iNumberX, iNumberY,
					 iNumIterations,
					 fDeltaP)
			  , m_Iflag(Iflag)
{
	// Allocate All needed memory
	cuICGN2D_prepare();
}


cuICGN2D::~cuICGN2D()
{}

void cuICGN2D::cuICGN2D_prepare()
{
	switch (m_Iflag)
	{
	case TW::paDIC::ICGN2DInterpolationFLag::Bicubic:
	{
		cudaMalloc((void**)&g_cuHandleICGN.m_d_fRx, sizeof(real_t)*m_iROIWidth*m_iROIHeight);
		cudaMalloc((void**)&g_cuHandleICGN.m_d_fRy, sizeof(real_t)*m_iROIWidth*m_iROIHeight);
		cudaMalloc((void**)&g_cuHandleICGN.m_d_fTx, sizeof(real_t)*m_iROIWidth*m_iROIHeight);
		cudaMalloc((void**)&g_cuHandleICGN.m_d_fTy, sizeof(real_t)*m_iROIWidth*m_iROIHeight);
		cudaMalloc((void**)&g_cuHandleICGN.m_d_fTxy, sizeof(real_t)*m_iROIWidth*m_iROIHeight);
		cudaMalloc((void**)&g_cuHandleICGN.m_d_f4InterpolationLUT, sizeof(real_t)*m_iROIWidth*m_iROIHeight * 4);

		break;
	}

	case TW::paDIC::ICGN2DInterpolationFLag::BicubicSpline:
	{
		cudaMalloc((void**)&g_cuHandleICGN.m_d_fRx, sizeof(real_t)*m_iROIWidth*m_iROIHeight);
		cudaMalloc((void**)&g_cuHandleICGN.m_d_fRy, sizeof(real_t)*m_iROIWidth*m_iROIHeight);
		cudaMalloc((void**)&g_cuHandleICGN.m_d_f4InterpolationLUT, sizeof(real_t)*m_iROIWidth*m_iROIHeight * 4);
		
		break;
	}

	default:
		break;
	}

	cudaMalloc((void**)&g_cuHandleICGN.m_d_fSubsetR, sizeof(real_t)*m_iPOINumber*(m_iSubsetSize));
	cudaMalloc((void**)&g_cuHandleICGN.m_d_fSubsetT, sizeof(real_t)*m_iPOINumber*(m_iSubsetSize));
	cudaMalloc((void**)&g_cuHandleICGN.m_d_fSubsetAveR, sizeof(real_t)*m_iPOINumber*(m_iSubsetSize + 1));
	cudaMalloc((void**)&g_cuHandleICGN.m_d_fSubsetAveT, sizeof(real_t)*m_iPOINumber*(m_iSubsetSize + 1));
	cudaMalloc((void**)&g_cuHandleICGN.m_d_Hessian, sizeof(real_t)*m_iPOINumber * 6 * 6);
	cudaMalloc((void**)&g_cuHandleICGN.m_d_Hessian, sizeof(real_t)*m_iPOINumber * m_iSubsetSize * 6);
	cudaMalloc((void**)&g_cuHandleICGN.m_d_iIterationNums, sizeof(int)*m_iPOINumber);
}

}// namespace paDIC
}// namespace TW
