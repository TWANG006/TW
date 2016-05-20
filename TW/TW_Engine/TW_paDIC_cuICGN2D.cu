#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#include "TW_utils.h"

#include "TW_paDIC_cuICGN2D.h"

namespace TW{
namespace paDIC{

#define BLOCK_SIZE_64  64
#define BLOCK_SIZE_128 128
#define BLOCK_SIZE_256 256

// !-----------------------------CUDA Kernel Functions -----------------------------------

/// \brief Kernel to compute sqrt[Sigma_i(R_i-R_m)^2] and construct all subsets for the refencence
/// ROI in the reference image
/// Note: This is only needed when isRefImgUpdated is true
/// 
/// \param	d_refImg image
/// \param	d_iPOIXY POI positions
/// \param iSubsetX, iSubsetY Half of the ROI size
/// \param iImgWidth, iImgHeight Image width&height of the d_refImg
/// \param d_wholeSubset output the constructed subsets iPOINum * iSubsetW * iSubsetH
/// \param d_wholeSubsetNorm iPOINum*(iSubsetW * iSubsetH +1) stores all the R_i - R_m and sqrt[Sigma_i(R_i-R_m)^2]

__global__ void RefAllSubetsNorm_Kernel(// Inputs
									    const uchar1 *d_refImg,
									    const int *d_iPOIXY,
									    const int iSubsetX, const int iSubsetY,
									    const int iImgWidth, const int iImgHeight,
									    //const int iStartX, const int iStartY,
									    //const int iROIWidth, const int iROIHeight,
									    // Outputs
									    real_t *d_wholeSubset,
									    real_t *d_wholeSubsetNorm)
{
	__shared__ real_t sm[BLOCK_SIZE_64];
	const int tid = threadIdx.x;
	const int dim = blockDim.x;
	const int bid = blockIdx.x;
	const int iSubsetW = iSubsetX * 2 + 1;
	const int iSubsetH = iSubsetY * 2 + 1;
	const int iSubsetSize = iSubsetW * iSubsetH;
	real_t avg = 0;// = 0;
	real_t mySum = 0;
	real_t tempt = 0;
	real_t *dSubSet = d_wholeSubset + iSubsetSize*bid;
	real_t *dSubsetAve = d_wholeSubsetNorm + (iSubsetSize + 1)*bid;

	// Construct refImgR and compute the mean value R_m
	for (int id = tid; id < iSubsetSize; id += dim)
	{
		int l = id / iSubsetW;	// y
		int m = id % iSubsetW;	// x

		tempt = (real_t)d_refImg[(d_iPOIXY[bid * 2 + 0] - iSubsetY + l)*iImgWidth + d_iPOIXY[bid * 2 + 1] - iSubsetX + m].x;
		dSubSet[id] = tempt;
		mySum += tempt / real_t(iSubsetSize);
	}
	__syncthreads();	
	reduceBlock<BLOCK_SIZE_64, real_t>(sm, mySum, tid);
	__syncthreads();

	// Calculate Sigma_i sqrt(R_i - R_m) 
	avg = sm[0];	// Norm
	mySum = 0; 
	for (int id = tid; id < iSubsetSize; id += dim)
	{
		tempt = dSubSet[id] - avg;	
		mySum += tempt * tempt;
		dSubsetAve[id + 1] = tempt;
	}
	__syncthreads();
	reduceBlock<BLOCK_SIZE_64, real_t>(sm, mySum, tid);
	__syncthreads();

	if(tid ==0)
	{
		dSubsetAve[0] = sqrt(sm[tid]);
	}
}

/// \brief
///
/// \param 
__global__ void InverseHessian_Kernel(real_t* d_Rx, 
									  real_t *d_Ry,
									  int_t *d_iPOIXY,
									  int_t iSubsetX,  int_t iSubsetY,
									  int_t iStartX, int_t iStartY,
									  int_t iROIWidth, int_t iROIHeight,
									  real_t2 *whole_d_RDescent,
									  real_t* whole_d_InvHessian)
{
	__shared__ real_t Hessian[96];
	__shared__ real_t sm[BLOCK_SIZE_64];
	__shared__ int iIndOfRowTempt[8];

	const int tid = threadIdx.x;
	const int dim = blockDim.x;
	const int bid = blockIdx.x;

	const int iSubsetW = iSubsetX * 2 + 1;
	const int iSubsetH = iSubsetY * 2 + 1;
	const int iSubsetSize = iSubsetW * iSubsetH;


	real_t tempt;
	real_t t_dD0, t_dD1, t_dD2, t_dD3, t_dD4, t_dD5;

	real_t2 *RDescent = whole_d_RDescent + bid * iSubsetSize * 3;
	real_t *r_InvHessian = whole_d_InvHessian + bid * 36;

	for (int id = tid; id < 96; id += dim)
	{
		Hessian[id] = 0;
	}

	for (int id = tid; id < iSubsetSize; id += dim)
	{
		int l = id / iSubsetW;
		int m = id % iSubsetW;
		
		real_t tx = d_Rx[(d_iPOIXY[bid * 2 + 0] - iSubsetY + l)*iROIWidth + d_iPOIXY[bid * 2 + 1] - iSubsetX + m];
		RDescent[l*iSubsetW + m].x = t_dD0 = tx;
		RDescent[l*iSubsetW + m].y = t_dD1 = tx * (m - iSubsetX);
		RDescent[iSubsetSize + l * iSubsetW + m].x = t_dD2 = tx * (l - iSubsetY);

		real_t ty = d_Ry[(d_iPOIXY[bid * 2 + 0] - iSubsetY + l)*iROIWidth + d_iPOIXY[bid * 2 + 1] - iSubsetX + m];
		RDescent[iSubsetSize + l * iSubsetW + m].y = t_dD3 = ty;
		RDescent[iSubsetSize * 2 + l * iSubsetW + m].x = t_dD4 = ty * (m - iSubsetX);
		RDescent[iSubsetSize * 2 + l * iSubsetW + m].y = t_dD5 = ty * (l - iSubsetY);

		//00		
		tempt=t_dD0 * t_dD0; 
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if(tid==0)
		{
			Hessian[0*16+0]+=sm[0];
		}
//11
		tempt=t_dD1 * t_dD1; 
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if(tid==0)
		{
			Hessian[1*16+1]+=sm[0];
		}
//22		
		tempt=t_dD2 * t_dD2; 
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if(tid==0)
		{
			Hessian[2*16+2]+=sm[0];
		}
//33
		tempt=t_dD3 * t_dD3; 
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if(tid==0)
		{
			Hessian[3*16+3]+=sm[0];
		}
//44		
		tempt=t_dD4 * t_dD4; 
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if(tid==0)
		{
			Hessian[4*16+4]+=sm[0];
		}
//55
		tempt=t_dD5 * t_dD5; 
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if(tid==0)
		{
			Hessian[5*16+5]+=sm[0];
		}


//01		
		tempt=t_dD0 * t_dD1; 
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if(tid==0)
		{
			Hessian[0*16+1]+=sm[0];
		}
//02
		tempt=t_dD0 * t_dD2; 
		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
		if(tid==0)
		{
			Hessian[0*16+2]+=sm[0];
		}
//03		
		tempt=t_dD0 * t_dD3; 
		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
		if(tid==0)
		{
			Hessian[0*16+3]+=sm[0];
		}
//04
		tempt=t_dD0 * t_dD4; 
		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
		if(tid==0)
		{
			Hessian[0*16+4]+=sm[0];
		}
//05		
		tempt=t_dD0 * t_dD5; 
		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
		if(tid==0)
		{
			Hessian[0*16+5]+=sm[0];
		}
//12
		tempt=t_dD1 * t_dD2; 
		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
		if(tid==0)
		{
			Hessian[1*16+2]+=sm[0];
		}
//13		
		tempt=t_dD1 * t_dD3; 
		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
		if(tid==0)
		{
			Hessian[1*16+3]+=sm[0];
		}
//14
		tempt=t_dD1 * t_dD4; 
		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
		if(tid==0)
		{
			Hessian[1*16+4]+=sm[0];
		}
//15		
		tempt=t_dD1 * t_dD5; 
		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
		if(tid==0)
		{
			Hessian[1*16+5]+=sm[0];
		}



//23		
		tempt=t_dD2 * t_dD3; 
		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
		if(tid==0)
		{
			Hessian[2*16+3]+=sm[0];
		}
//24
		tempt=t_dD2 * t_dD4; 
		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
		if(tid==0)
		{
			Hessian[2*16+4]+=sm[0];
		}
//25		
		tempt=t_dD2 * t_dD5; 
		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
		if(tid==0)
		{
			Hessian[2*16+5]+=sm[0];
		}


//34
		tempt=t_dD3 * t_dD4; 
		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
		if(tid==0)
		{
			Hessian[3*16+4]+=sm[0];
		}
//35		
		tempt=t_dD3 * t_dD5; 
		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
		if(tid==0)
		{
			Hessian[3*16+5]+=sm[0];
		}

//45		
		tempt=t_dD4 * t_dD5; 
		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
		if(tid==0)
		{
			Hessian[4*16+5]+=sm[0];
		}
		__syncthreads();
		if (tid < BLOCK_SIZE_64)
			sm[tid] = 0;
	}

	if(tid<5)
	{
		Hessian[(tid + 1) * 16 + 0] = Hessian[0 * 16 + (tid + 1)];
	}
	if(tid<4)
	{
		Hessian[(tid + 2) * 16 + 1] = Hessian[1 * 16 + (tid + 2)];
	}
	if(tid<3)
	{
		Hessian[(tid + 3) * 16 + 2] = Hessian[2 * 16 + (tid + 3)];
	}
	if(tid<2)
	{
		Hessian[(tid + 4) * 16 + 3] = Hessian[3 * 16 + (tid + 4)];
	}
	if(tid==0)
	{
		Hessian[5 * 16 + 4] = Hessian[4 * 16 + 5];
	}

	// Initialize the Hessian matrix 
	if(tid<6)
	{
		Hessian[tid * 16 + tid + 8] = 1;
	}

	// Pivoting
	if (tid < 16)
	{
		for (int l = 0; l < 6; l++)
		{
			//Find pivot (maximum lth column element) in the rest (6-l) rows
			//找到最大值
			if (tid < 8)
			{
				iIndOfRowTempt[tid] = l;
			}
			if (tid < 6 - l)
			{
				iIndOfRowTempt[tid] = tid + l;
			}
			if (tid < 4)
			{
				if (Hessian[iIndOfRowTempt[tid] * 16 + l] < Hessian[iIndOfRowTempt[tid + 4] * 16 + l])
					iIndOfRowTempt[tid] = iIndOfRowTempt[tid + 4];
			}
			if (tid < 2)
			{
				if (Hessian[iIndOfRowTempt[tid] * 16 + l] < Hessian[iIndOfRowTempt[tid + 2] * 16 + l])
					iIndOfRowTempt[tid] = iIndOfRowTempt[tid + 2];
			}
			if (tid == 0)
			{
				if (Hessian[iIndOfRowTempt[tid] * 16 + l] < Hessian[iIndOfRowTempt[tid + 1] * 16 + l])
					iIndOfRowTempt[tid] = iIndOfRowTempt[tid + 1];
				
				// Maximum's ind is stored in iIndOfRowTempt[0]
				if (Hessian[iIndOfRowTempt[tid] * 16 + l] < 0.0000001)
				{
					Hessian[iIndOfRowTempt[tid] * 16 + l] = 0.0000001;
				}
			}
			if (tid < 12)
			{
				int m_iIndexOfCol = tid / 6 * 8 + tid % 6;
				float m_dTempt;
				//swap 操作
				if (iIndOfRowTempt[0] != l)
				{
					m_dTempt = Hessian[iIndOfRowTempt[0] * 16 + m_iIndexOfCol];
					Hessian[iIndOfRowTempt[0] * 16 + m_iIndexOfCol] = Hessian[l * 16 + m_iIndexOfCol];
					Hessian[l * 16 + m_iIndexOfCol] = m_dTempt;
				}


				// Perform row operation to form required identity matrix out of the Hessian matrix
				Hessian[l * 16 + m_iIndexOfCol] /= Hessian[l * 16 + l];

				for (int next_row = 0; next_row < 6; next_row++)
				{
					if (next_row != l)
					{
						Hessian[next_row * 16 + m_iIndexOfCol] -= Hessian[l * 16 + m_iIndexOfCol] * Hessian[next_row * 16 + l];
					}
				}
			}
		}
	}


	//inv Hessian
	if (tid < 32)
		r_InvHessian[tid] = Hessian[tid / 6 * 16 + tid % 6 + 8];
	if (tid < 4)
	{
		//	tid+=32;
		r_InvHessian[tid + 32] = Hessian[(tid + 32) / 6 * 16 + (tid + 32) % 6 + 8];
	}
}


// ------------------------------CUDA Kernel Functions End-------------------------------!


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
			  , m_isRefImgUpdated(false)
{
	// Allocate All needed memory
	prepare();
}


cuICGN2D::~cuICGN2D()
{}

void cuICGN2D::prepare()
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
		cudaMalloc((void**)&g_cuHandleICGN.m_d_f4InterpolationLUT, sizeof(real_t4) * 4 * m_iROIWidth*m_iROIHeight);

		break;
	}

	case TW::paDIC::ICGN2DInterpolationFLag::BicubicSpline:
	{
		cudaMalloc((void**)&g_cuHandleICGN.m_d_fRx, sizeof(real_t)*m_iROIWidth*m_iROIHeight);
		cudaMalloc((void**)&g_cuHandleICGN.m_d_fRy, sizeof(real_t)*m_iROIWidth*m_iROIHeight);
		cudaMalloc((void**)&g_cuHandleICGN.m_d_f4InterpolationLUT, sizeof(real_t4) * 4 * m_iROIWidth*m_iROIHeight);
		
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
	cudaMalloc((void**)&g_cuHandleICGN.m_d_RDescent, sizeof(real_t2) * 3 * m_iPOINumber * m_iSubsetSize);
	cudaMalloc((void**)&g_cuHandleICGN.m_d_iIterationNums, sizeof(int)*m_iPOINumber);
}

void cuICGN2D::cuCompute(uchar1 *d_fTarImg,
						 int_t  *d_iPOIXY,
						 real_t *d_fU,
						 real_t *d_fV)
{
	g_cuHandleICGN.m_d_fTarImg = d_fTarImg;
	g_cuHandleICGN.m_d_iPOIXY = d_iPOIXY;
	g_cuHandleICGN.m_d_fU = d_fU;
	g_cuHandleICGN.m_d_fV = d_fV;


}

void cuICGN2D::cuInitialize(uchar1 *d_fRefImg)
{
	g_cuHandleICGN.m_d_fRefImg = d_fRefImg;
	m_isRefImgUpdated = true;
}

void cuICGN2D::Initialize(cv::Mat& refImg)
{
	// TODO
}

}// namespace paDIC
}// namespace TW
