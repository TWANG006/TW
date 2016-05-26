#include <QDebug>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#include "TW_utils.h"
#include "TW_MemManager.h"

#include "TW_paDIC_cuICGN2D.h"

namespace TW{
namespace paDIC{

#define BLOCK_SIZE_64  64
#define BLOCK_SIZE_128 128
#define BLOCK_SIZE_256 256

// !-------------------------------------------CUDA Kernel Functions ---------------------------------------------!

/// \brief Kernel to compute sqrt[Sigma_i(R_i-R_m)^2] and construct all subsets for the refencence
/// ROI in the reference image
/// Note: This is only needed when isRefImgUpdated is true
/// 
/// \param d_refImg image
/// \param d_iPOIXY POI positions
/// \param iSubsetX, iSubsetY Half of the ROI size
/// \param iImgWidth, iImgHeight Image width&height of the d_refImg
/// \param d_wholeSubset output the constructed subsets iPOINum * iSubsetW * iSubsetH
/// \param d_wholeSubsetNorm iPOINum*(iSubsetW * iSubsetH +1) stores all the R_i - R_m and sqrt[Sigma_i(R_i-R_m)^2]
__global__ void RefAllSubetsNorm_Kernel(// Inputs
								  	    const uchar1*d_refImg, 
										const int *d_iPOIXY,
										const int iSubsetW, const int iSubsetH,
										const int iSubsetX, const int iSubsetY,
										const int iImgWidth, const int m_iImgHeight,
										// Outputs
										real_t *whole_dSubSet,
										real_t *whole_dSubsetAve)
{
	//默认smsize和dim大小相等
	//dim取64
	__shared__ real_t sm[BLOCK_SIZE_64];
	int tid = threadIdx.x;
	int dim = blockDim.x;
	int bid = blockIdx.x;
	int size = iSubsetH * iSubsetW;
	real_t avg;// = 0;
	real_t mySum = 0;
	real_t tempt;
	real_t *fSubSet = whole_dSubSet + size*bid;
	real_t *fSubsetAve = whole_dSubsetAve + (size + 1)*bid;
	for (int id = tid; id < size; id += dim)
	{
		int	l = id / iSubsetW;
		int m = id % iSubsetW;
		tempt = (real_t)d_refImg[int(d_iPOIXY[bid * 2] - iSubsetY + l)*iImgWidth + int(d_iPOIXY[bid * 2 + 1] - iSubsetX + m)].x;
		fSubSet[id] = tempt;
		mySum += tempt / size;
	}
	__syncthreads();
	reduceBlock<BLOCK_SIZE_64, float>(sm, mySum, tid);
	__syncthreads();
	avg = sm[0];
	mySum = 0;
	for (int id = tid; id < size; id += dim)
	{
		tempt = fSubSet[id] - avg;
		mySum += tempt*tempt;
		fSubsetAve[id + 1] = tempt;
	}
	__syncthreads();
	reduceBlock<BLOCK_SIZE_64, float>(sm, mySum, tid);
	__syncthreads();
	if (tid == 0)
	{
		fSubsetAve[0] = sqrt(sm[tid]);
	}
}

/// \brief Compute the RDescent vector and the inverse Hessian matrix 
/// using Gaussian Jordan Elimination algorithm
///
/// \param d_Rx iROIWidth * iROIHeight, GradientX of the refImg
/// \param d_Ry iROIWidth * iROIHeight, GradientY of the refImg
/// \param d_iPOIXY m_iPOINum * 2 POI positions within the images
/// \param iSubsetX, iSubsetY Half of the subset size
/// \param iSubsetW, iSubsetH Subset Size
/// \param iROIWidth, iROIHeight ROI size
/// \param whole_d_RDescent The RDescent vector for all POI within the refImg
/// \param whole_d_InvHessian The inverse of the Hessian matrix for all POIs within the refImg
__global__ void InverseHessian_Kernel(// Inputs
									  const real_t* d_Rx, 
									  const real_t *d_Ry,
									  const int *d_iPOIXY,
									  const int iSubsetX, const int iSubsetY,
									  const int iSubsetW, const int iSubsetH,
									  const int iStartX, const int iStartY,
									  const int iROIWidth, const int iROIHeight,
									  // Outputs
									  real_t2 *whole_d_RDescent,
									  float* whole_d_InvHessian)
{
	__shared__ real_t Hessian[96];
	__shared__ real_t sm[BLOCK_SIZE_64];
	__shared__ int iIndOfRowTempt[8];
	int tid = threadIdx.x;
	int dim = blockDim.x;
	int bid = blockIdx.x;
	int iSubWindowSize = iSubsetH * iSubsetW;
	int l;
	int m;

	real_t tempt;
	real_t t_dD0;
	real_t t_dD1;
	real_t t_dD2;
	real_t t_dD3;
	real_t t_dD4;
	real_t t_dD5;

	real_t2* RDescent = whole_d_RDescent + bid*iSubWindowSize * 3;
	real_t *r_dInvHessian = whole_d_InvHessian + bid * 36;
	for (int id = tid; id < 96; id += dim)
	{
		Hessian[id] = 0;
	}

	for (int id = tid; id < iSubWindowSize; id += dim)
	{
		l = id / iSubsetW;
		m = id % iSubsetW;
		real_t tx = d_Rx[iROIWidth*(d_iPOIXY[bid * 2] - iSubsetY + l - iStartY) + d_iPOIXY[bid * 2 + 1] - iSubsetX + m - iStartX];
		RDescent[l*iSubsetW + m].x = t_dD0 = tx;
		RDescent[l*iSubsetW + m].y = t_dD1 = tx*(m - iSubsetX);
		RDescent[iSubWindowSize + l*iSubsetW + m].x = t_dD2 = tx*(l - iSubsetY);

		real_t ty = d_Ry[iROIWidth*(d_iPOIXY[bid * 2] - iSubsetY + l - iStartY) + d_iPOIXY[bid * 2 + 1] - iSubsetX + m - iStartX];
		RDescent[iSubWindowSize + l*iSubsetW + m].y = t_dD3 = ty;
		RDescent[iSubWindowSize * 2 + l*iSubsetW + m].x = t_dD4 = ty*(m - iSubsetX);
		RDescent[iSubWindowSize * 2 + l*iSubsetW + m].y = t_dD5 = ty*(l - iSubsetY);
		//00		
		tempt = t_dD0 * t_dD0;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[0 * 16 + 0] += sm[0];
		}
		//11
		tempt = t_dD1 * t_dD1;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[1 * 16 + 1] += sm[0];
		}
		//22		
		tempt = t_dD2 * t_dD2;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[2 * 16 + 2] += sm[0];
		}
		//33
		tempt = t_dD3 * t_dD3;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[3 * 16 + 3] += sm[0];
		}
		//44		
		tempt = t_dD4 * t_dD4;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[4 * 16 + 4] += sm[0];
		}
		//55
		tempt = t_dD5 * t_dD5;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[5 * 16 + 5] += sm[0];
		}


		//01		
		tempt = t_dD0 * t_dD1;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[0 * 16 + 1] += sm[0];
		}
		//02
		tempt = t_dD0 * t_dD2;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[0 * 16 + 2] += sm[0];
		}
		//03		
		tempt = t_dD0 * t_dD3;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[0 * 16 + 3] += sm[0];
		}
		//04
		tempt = t_dD0 * t_dD4;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[0 * 16 + 4] += sm[0];
		}
		//05		
		tempt = t_dD0 * t_dD5;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[0 * 16 + 5] += sm[0];
		}




		//12
		tempt = t_dD1 * t_dD2;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[1 * 16 + 2] += sm[0];
		}
		//13		
		tempt = t_dD1 * t_dD3;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[1 * 16 + 3] += sm[0];
		}
		//14
		tempt = t_dD1 * t_dD4;
		reduceBlock<BLOCK_SIZE_64, float>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[1 * 16 + 4] += sm[0];
		}
		//15		
		tempt = t_dD1 * t_dD5;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[1 * 16 + 5] += sm[0];
		}



		//23		
		tempt = t_dD2 * t_dD3;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[2 * 16 + 3] += sm[0];
		}
		//24
		tempt = t_dD2 * t_dD4;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[2 * 16 + 4] += sm[0];
		}
		//25		
		tempt = t_dD2 * t_dD5;
		reduceBlock<BLOCK_SIZE_64, float>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[2 * 16 + 5] += sm[0];
		}


		//34
		tempt = t_dD3 * t_dD4;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[3 * 16 + 4] += sm[0];
		}
		//35		
		tempt = t_dD3 * t_dD5;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[3 * 16 + 5] += sm[0];
		}

		//45		
		tempt = t_dD4 * t_dD5;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
		if (tid == 0)
		{
			Hessian[4 * 16 + 5] += sm[0];
		}
		/*if((tid+dim<m_iSubWindowSize))
		{	*/
		__syncthreads();
		if (tid < BLOCK_SIZE_64)
			sm[tid] = 0;
		//}
	}
	if (tid < 5)
	{
		Hessian[(tid + 1) * 16 + 0] = Hessian[0 * 16 + (tid + 1)];
	}
	if (tid < 4)
	{
		Hessian[(tid + 2) * 16 + 1] = Hessian[1 * 16 + (tid + 2)];
	}
	if (tid < 3)
	{
		Hessian[(tid + 3) * 16 + 2] = Hessian[2 * 16 + (tid + 3)];
	}
	if (tid < 2)
	{
		Hessian[(tid + 4) * 16 + 3] = Hessian[3 * 16 + (tid + 4)];
	}
	if (tid == 0)
	{
		Hessian[5 * 16 + 4] = Hessian[4 * 16 + 5];
	}

	//初始化inv数组
	if (tid < 6)
	{
		Hessian[tid * 16 + tid + 8] = 1;
	}
	//这里展开吧

	//6
	//find the max
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
				//最大值的ind在iIndOfRowTempt[0]中
				//以下为健壮性处理
				if (Hessian[iIndOfRowTempt[tid] * 16 + l] == 0)
				{
					Hessian[iIndOfRowTempt[tid] * 16 + l] = 0.0000001;
				}
			}
			if (tid < 12)
			{
				int m_iIndexOfCol = tid / 6 * 8 + tid % 6;
				real_t m_dTempt;
				//swap 操作
				if (iIndOfRowTempt[0] != l)
				{
					m_dTempt = Hessian[iIndOfRowTempt[0] * 16 + m_iIndexOfCol];
					Hessian[iIndOfRowTempt[0] * 16 + m_iIndexOfCol] = Hessian[l * 16 + m_iIndexOfCol];
					Hessian[l * 16 + m_iIndexOfCol] = m_dTempt;
				}


				// Perform row operation to form required identity matrix out of the Hessian matrix

				//把l行归一化
				Hessian[l * 16 + m_iIndexOfCol] /= Hessian[l * 16 + l];
				//每一行减一下
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
		r_dInvHessian[tid] = Hessian[tid / 6 * 16 + tid % 6 + 8];
	if (tid < 4)
	{
		//	tid+=32;
		r_dInvHessian[tid + 32] = Hessian[(tid + 32) / 6 * 16 + (tid + 32) % 6 + 8];
	}
}

/// \brief Parallel ICGN algorithm. The deformation parameter P = [u, ux, uy, v, vx, vy] is computed by this method.
/// Also, the displacements d_fU & d_fV are also updated to their sub-pixel precision.
///
/// \param d_dPXY POI positions
/// \param iImgWidth, iImgHeight Image size
/// \param iStartX, iStartY Start position of the ROI
/// \param iROIWidth, iROIHeight ROI size
/// \param iSubsetX, iSubsetY Half of the subset size
/// \param iSubsetW, iSubsetH Subset size 
/// \param iMaxIteration Maxmum iteration number for the iterations
/// \param fDeltaP The convergence creterion
/// \param d_tarImg The target image
/// \param whole_d_dInvHessian The inverse of the Hessian matrix of whole POIs within the refImg
/// \param fInterpolation The interpolation LUT 
/// \param whole_d_2dRDecent The RDescent vectors of all POIs within the refImg
/// \param whole_d_dSubsetAveR The R_i - R_m of all Subsets within the refImg
/// \param whole_d_dSubsetT The whole subsets constructed for the tarImg
/// \param whole_d_dSubsetAveT The T_i - T_m within the tarImg
/// \param whole_d_iIteration The number of iterations used by each POI
/// \param whole_d_dP The deformation parameter P calculated for all the POIs 
__global__ void ICGN_Computation_Kernel(// Inputs
										real_t* d_fU, float *d_fV,
										const int* d_iPOIXY,
										const int iImgWidth, const int iImgHeight,
										const int iStartX, const int iStartY,
										const int iROIWidth, const int iROIHeight,
										const int iSubsetX, const int iSubsetY,
										const int iSubsetW, const int iSubsetH,
										const int iMaxIteration,
										const real_t fDeltaP,
										const uchar1 *d_tarImg,
										real_t*whole_d_dInvHessian,
										real_t4 *m_dTBicubic,
										real_t2 *whole_d_2dRDescent,
										real_t *whole_d_dSubsetAveR,
										// Tempts
										real_t*whole_d_dSubsetT,
										real_t *whole_d_dSubsetAveT,
										// Outputs
										int *whole_d_iIteration,
										real_t *whole_d_dP)
{
	__shared__ real_t sm[BLOCK_SIZE_64];
	__shared__ real_t DP[6];
	__shared__ real_t Warp[6];
	__shared__ real_t P[6];
	__shared__ int break_sig[1];

	int tid = threadIdx.x;
	int dim = blockDim.x;
	int bid = blockIdx.x;

	real_t fWarpX, fWarpY;
	int iTempX, iTempY;
	real_t fTempX, fTempY;
	real_t ftemptVal;//=0;
	int size = iSubsetH*iSubsetW;

	real_t *fSubsetT = whole_d_dSubsetT + iSubsetH*iSubsetW*bid;
	real_t *fSubsetAveT = whole_d_dSubsetAveT + (iSubsetH*iSubsetW + 1)*bid;
	real_t *fInvHessian = whole_d_dInvHessian + bid * 36;
	real_t2 *fRDescent = whole_d_2dRDescent + bid*iSubsetH*iSubsetW * 3;
	real_t *fSubsetAveR = whole_d_dSubsetAveR + bid*(iSubsetH*iSubsetW + 1);

	if (tid == 0)
	{
		// Transfer the initial guess to IC-GN algorithm
		P[0] = d_fU[bid];
		P[1] = 0;
		P[2] = 0;
		P[3] = d_fV[bid];
		P[4] = 0;
		P[5] = 0;

		// Initialize the warp matrix
		Warp[0] = 1 + P[1];
		Warp[1] = P[2];
		Warp[2] = P[0];
		Warp[3] = P[4];
		Warp[4] = 1 + P[5];
		Warp[5] = P[3];
	}
	if (tid == 32)
	{
		break_sig[0] = 0;
	}
	__syncthreads();
	int iIteration;
	for (iIteration = 0; iIteration < iMaxIteration; iIteration++)
	{
		real_t mySum = 0;
		for (int id = tid; id < size; id += dim)
		{
			int l = id / iSubsetW;
			int m = id % iSubsetW;
			if (l < iSubsetH && m < iSubsetW)
			{
				fWarpX = d_iPOIXY[2 * bid + 1] + Warp[0] * (m - iSubsetX) + Warp[1] * (l - iSubsetY) + Warp[2];
				fWarpY = d_iPOIXY[2 * bid + 0] + Warp[3] * (m - iSubsetX) + Warp[4] * (l - iSubsetY) + Warp[5];
				
				if (fWarpX < iStartX) fWarpX = iStartX;
				if (fWarpY < iStartY) fWarpY = iStartY;
				if (fWarpX >= iROIWidth + iStartX)  fWarpX = iROIWidth + iStartX - 1;
				if (fWarpY >= iROIHeight + iStartY) fWarpY = iROIHeight + iStartY - 1;

				iTempX = int(fWarpX);
				iTempY = int(fWarpY);

				fTempX = fWarpX - iTempX;
				fTempY = fWarpY - iTempY;
				if ((fTempX <= 0.0000001) && (fTempY == 0.0000001))
				{
					ftemptVal = (real_t)d_tarImg[iTempY * iImgWidth + iTempX].x;
				}
				else
				{
					//unroll for loop
					real_t4 a1, a2, a3, a4;
					a1 = m_dTBicubic[0 * iROIWidth*iROIHeight + (iTempY - iStartY)*iROIWidth + iTempX - iStartX];
					a2 = m_dTBicubic[1 * iROIWidth*iROIHeight + (iTempY - iStartY)*iROIWidth + iTempX - iStartX];
					a3 = m_dTBicubic[2 * iROIWidth*iROIHeight + (iTempY - iStartY)*iROIWidth + iTempX - iStartX];
					a4 = m_dTBicubic[3 * iROIWidth*iROIHeight + (iTempY - iStartY)*iROIWidth + iTempX - iStartX];

					ftemptVal =
						a1.w * pow(fTempY, 0) * pow(fTempX, 0) +
						a1.x * pow(fTempY, 0) * pow(fTempX, 1) +
						a1.y * pow(fTempY, 0) * pow(fTempX, 2) +
						a1.z * pow(fTempY, 0) * pow(fTempX, 3) +

						a2.w * pow(fTempY, 1) * pow(fTempX, 0) +
						a2.x * pow(fTempY, 1) * pow(fTempX, 1) +
						a2.y * pow(fTempY, 1) * pow(fTempX, 2) +
						a2.z * pow(fTempY, 1) * pow(fTempX, 3) +

						a3.w * pow(fTempY, 2) * pow(fTempX, 0) +
						a3.x * pow(fTempY, 2) * pow(fTempX, 1) +
						a3.y * pow(fTempY, 2) * pow(fTempX, 2) +
						a3.z * pow(fTempY, 2) * pow(fTempX, 3) +

						a4.w * pow(fTempY, 3) * pow(fTempX, 0) +
						a4.x * pow(fTempY, 3) * pow(fTempX, 1) +
						a4.y * pow(fTempY, 3) * pow(fTempX, 2) +
						a4.z * pow(fTempY, 3) * pow(fTempX, 3);

				}
				fSubsetT[l*iSubsetW + m] = ftemptVal;
				mySum += ftemptVal / size;
			}
		}

		__syncthreads();
		real_t avg;// = 0;
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, mySum, tid);
		__syncthreads();
		avg = sm[0];
		mySum = 0;
		for (int id = tid; id < size; id += dim)
		{
			ftemptVal = fSubsetT[id] - avg;
			mySum += ftemptVal * ftemptVal;
			fSubsetAveT[id + 1] = ftemptVal;
		}
		__syncthreads();
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, mySum, tid);
		__syncthreads();

		if (tid == 0)
		{

			fSubsetAveT[0] = ftemptVal = sqrt(sm[tid]);
			sm[tid] = fSubsetAveR[0] / ftemptVal;
		}

		real_t n0, n1, n2, n3, n4, n5;
		n0 = 0; n1 = 0; n2 = 0; n3 = 0; n4 = 0; n5 = 0;
		real_t2 rd;
		__syncthreads();
		real_t Nor = sm[0];//m_dSubNorR[0]/m_dSubNorT[0];
		//	__syncthreads();
		for (int id = tid; id < size; id += dim)
		{
			//		int l=id/m_iSubsetW;
			//		int m=id%m_iSubsetW;
			ftemptVal = (Nor)* fSubsetAveT[id + 1] - fSubsetAveR[id + 1];
			rd = fRDescent[id];
			n0 += (rd.x * ftemptVal);
			n1 += (rd.y * ftemptVal);
			rd = fRDescent[size + id];
			n2 += (rd.x * ftemptVal);
			n3 += (rd.y * ftemptVal);
			rd = fRDescent[size * 2 + id];
			n4 += (rd.x * ftemptVal);
			n5 += (rd.y * ftemptVal);
		}


		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n0, tid);
		//	__syncthreads();
		if (tid < 6)
			n0 = sm[0];
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n1, tid);
		//	__syncthreads();
		if (tid < 6)
			n1 = sm[0];
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n2, tid);
		//	__syncthreads();
		if (tid < 6)
			n2 = sm[0];
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n3, tid);
		//	__syncthreads();
		if (tid < 6)
			n3 = sm[0];
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n4, tid);
		//	__syncthreads();
		if (tid < 6)
			n4 = sm[0];
		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n5, tid);
		//	__syncthreads();
		if (tid < 6)
			n5 = sm[0];
		if (tid < 6)
		{
			DP[tid] =
				fInvHessian[tid * 6 + 0] * n0 +
				fInvHessian[tid * 6 + 1] * n1 +
				fInvHessian[tid * 6 + 2] * n2 +
				fInvHessian[tid * 6 + 3] * n3 +
				fInvHessian[tid * 6 + 4] * n4 +
				fInvHessian[tid * 6 + 5] * n5;
		}

		if (tid == 0)
		{
			ftemptVal = (1 + DP[1]) * (1 + DP[5]) - DP[2] * DP[4];
			Warp[0] = ((1 + P[1]) * (1 + DP[5]) - P[2] * DP[4]) / ftemptVal;
			Warp[1] = (P[2] * (1 + DP[1]) - (1 + P[1]) * DP[2]) / ftemptVal;
			Warp[2] = P[0] + (P[2] * (DP[0] * DP[4] - DP[3] - DP[3] * DP[1]) - (1 + P[1]) * (DP[0] * DP[5] + DP[0] - DP[2] * DP[3])) / ftemptVal;
			Warp[3] = (P[4] * (1 + DP[5]) - (1 + P[5]) * DP[4]) / ftemptVal;
			Warp[4] = ((1 + P[5]) * (1 + DP[1]) - P[4] * DP[2]) / ftemptVal;
			Warp[5] = P[3] + ((1 + P[5]) * (DP[0] * DP[4] - DP[3] - DP[3] * DP[1]) - P[4] * (DP[0] * DP[5] + DP[0] - DP[2] * DP[3])) / ftemptVal;

			// Update DeltaP
			P[0] = Warp[2];
			P[1] = Warp[0] - 1;
			P[2] = Warp[1];
			P[3] = Warp[5];
			P[4] = Warp[3];
			P[5] = Warp[4] - 1;

			if (sqrt(
				pow(DP[0], 2) +
				pow(DP[1] * iSubsetX, 2) +
				pow(DP[2] * iSubsetY, 2) +
				pow(DP[3], 2) +
				pow(DP[4] * iSubsetX, 2) +
				pow(DP[5] * iSubsetY, 2))
				< fDeltaP)
			{
				break_sig[0] = 1;
			}
		}
		__syncthreads();
		if (break_sig[0] == 1)
			break;
	}
	if (tid == 0)
	{
		whole_d_iIteration[bid] = iIteration;
		d_fV[bid] = P[3];
		d_fU[bid] = P[0];
	}
	if (tid < 6)
	{
		whole_d_dP[bid * 6 + tid] = P[tid];
	}
}







// ----------------------------------------------CUDA Kernel Functions End----------------------------------------!


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
{
	cuFinalize();

	cudaSafeFree(g_cuHandleICGN.m_d_fRefImg);	// Reference image
	cudaSafeFree(g_cuHandleICGN.m_d_fTarImg);	// Target image

	cudaSafeFree(g_cuHandleICGN.m_d_iPOIXY);	// POI positions on device
	cudaSafeFree(g_cuHandleICGN.m_d_fU);		// Displacement in x-direction
	cudaSafeFree(g_cuHandleICGN.m_d_fV);		// Displacement in y-direction
}

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
	cudaMalloc((void**)&g_cuHandleICGN.m_d_invHessian, sizeof(real_t)*m_iPOINumber * 6 * 6);
	cudaMalloc((void**)&g_cuHandleICGN.m_d_RDescent, sizeof(real_t2) * 3 * m_iPOINumber * m_iSubsetSize);
	cudaMalloc((void**)&g_cuHandleICGN.m_d_iIterationNums, sizeof(int)*m_iPOINumber);
	cudaMalloc((void**)&g_cuHandleICGN.m_d_dP, sizeof(real_t)*m_iPOINumber * 6);
}

void cuICGN2D::cuInitialize(uchar1 *d_fRefImg)
{
	g_cuHandleICGN.m_d_fRefImg = d_fRefImg;
	m_isRefImgUpdated = true;
}

void cuICGN2D::Initialize(cv::Mat& refImg)
{
	ResetRefImg(refImg);
}

void cuICGN2D::ResetRefImg(const cv::Mat& refImg) 
{
	if (refImg.cols != m_iImgWidth || refImg.rows != m_iImgHeight)
	{
		qDebug()<<"Image Dimension Noat Match";
		return;
	}

	cudaMemcpy(g_cuHandleICGN.m_d_fRefImg, 
			   (void*)refImg.data,
			   refImg.cols * refImg.rows,
			   cudaMemcpyHostToDevice);

	m_isRefImgUpdated = true;
}

void cuICGN2D::SetTarImg(const cv::Mat& tarImg)
{
	if (tarImg.cols != m_iImgWidth || tarImg.rows != m_iImgHeight)
	{
		qDebug()<<"Image Dimension Noat Match";
		return;
	}

	cudaMemcpy(g_cuHandleICGN.m_d_fRefImg, 
			   (void*)tarImg.data,
			   tarImg.cols * tarImg.rows,
			   cudaMemcpyHostToDevice);
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
	
	if(m_isRefImgUpdated)
	{
		switch (m_Iflag)
		{
		case TW::paDIC::ICGN2DInterpolationFLag::Bicubic:
		{
			cuGradientXY_2Images(g_cuHandleICGN.m_d_fRefImg,
								 g_cuHandleICGN.m_d_fTarImg,
								 m_iStartX, m_iStartY,
								 m_iROIWidth, m_iROIHeight,
								 m_iImgWidth, m_iImgHeight,
								 TW::AccuracyOrder::Quadratic,
								 g_cuHandleICGN.m_d_fRx,
								 g_cuHandleICGN.m_d_fRy,
								 g_cuHandleICGN.m_d_fTx,
								 g_cuHandleICGN.m_d_fTy,
								 g_cuHandleICGN.m_d_fTxy);

			cuBicubicCoefficients(g_cuHandleICGN.m_d_fTarImg, 
								  g_cuHandleICGN.m_d_fTx,
								  g_cuHandleICGN.m_d_fTy,
								  g_cuHandleICGN.m_d_fTxy,
								  m_iStartX, m_iStartY,
								  m_iROIWidth, m_iROIHeight,
								  m_iImgWidth, m_iImgHeight,
								  g_cuHandleICGN.m_d_f4InterpolationLUT);

			 //For debug
		/*	std::cout << "Bicubic First: " << std::endl;
			std::cout.precision(10);
			float4 *interpolation;
			hcreateptr(interpolation, 4 * m_iROIWidth*m_iROIHeight);
			cudaMemcpy(interpolation, g_cuHandleICGN.m_d_f4InterpolationLUT, sizeof(real_t4) * 4 * m_iROIWidth*m_iROIHeight,
				cudaMemcpyDeviceToHost);
			for (int i = 0; i < 4; i++)
			{
				std::cout << interpolation[i*m_iROIWidth*m_iROIHeight].w << ", ";
				std::cout << interpolation[i*m_iROIWidth*m_iROIHeight].x << ", ";
				std::cout << interpolation[i*m_iROIWidth*m_iROIHeight].y << ", ";
				std::cout << interpolation[i*m_iROIWidth*m_iROIHeight].z << ", ";
				std::cout << std::endl;
			}

			hdestroyptr(interpolation);*/

			break;
		}
		case TW::paDIC::ICGN2DInterpolationFLag::BicubicSpline:
		{
			// TODO
			break;
		}
		default:
			break;
		}

		RefAllSubetsNorm_Kernel<<<m_iPOINumber, BLOCK_SIZE_64>>>(g_cuHandleICGN.m_d_fRefImg,
															     g_cuHandleICGN.m_d_iPOIXY,
															     m_iSubsetW, m_iSubsetH,
															     m_iSubsetX, m_iSubsetY,
															     m_iImgWidth, m_iImgHeight,
															     g_cuHandleICGN.m_d_fSubsetR,
															     g_cuHandleICGN.m_d_fSubsetAveR);

		InverseHessian_Kernel<<<m_iPOINumber, BLOCK_SIZE_64>>>(g_cuHandleICGN.m_d_fRx,
															   g_cuHandleICGN.m_d_fRy,
															   g_cuHandleICGN.m_d_iPOIXY,
															   m_iSubsetX, m_iSubsetY,
															   m_iSubsetW, m_iSubsetH,
															   m_iStartX, m_iStartY,
															   m_iROIWidth, m_iROIHeight,
															   g_cuHandleICGN.m_d_RDescent,
															   g_cuHandleICGN.m_d_invHessian);	
		m_isRefImgUpdated = false;
	}


	// For debug
	//std::cout << "Hessian" << std::endl;
	//float *dHessian;
	//hcreateptr(dHessian, m_iPOINumber * 6 * 6);
	//cudaMemcpy(dHessian, g_cuHandleICGN.m_d_invHessian, sizeof(float)* m_iPOINumber * 6 * 6, cudaMemcpyDeviceToHost);

	//for (int i = 0; i < 6; i++)
	//{
	//	for (int j = 0; j < 6; j++)
	//	{
	//		std::cout << dHessian[i * 6 + j] << ", ";
	//	}
	//	std::cout << "\n";
	//}
	//hdestroyptr(dHessian);

	// For Debug
	/*real_t *test;
	hcreateptr(test, m_iPOINumber*(m_iSubsetSize + 1));
	cudaMemcpy(test, g_cuHandleICGN.m_d_fSubsetAveR,  sizeof(real_t)*m_iPOINumber*(m_iSubsetSize + 1), cudaMemcpyDeviceToHost);
	std::cout<<test[0]<<", "<<std::endl;
	hdestroyptr(test);*/

	ICGN_Computation_Kernel<<<m_iPOINumber, BLOCK_SIZE_64>>>(g_cuHandleICGN.m_d_fU,
															 g_cuHandleICGN.m_d_fV,
															 g_cuHandleICGN.m_d_iPOIXY,
															 m_iImgWidth, m_iImgHeight,
															 m_iStartX, m_iStartY,
															 m_iROIWidth, m_iROIHeight,
															 m_iSubsetX, m_iSubsetY,
															 m_iSubsetW, m_iSubsetH,
															 m_iNumIterations,
															 m_fDeltaP,
															 g_cuHandleICGN.m_d_fTarImg,
															 g_cuHandleICGN.m_d_invHessian,
															 g_cuHandleICGN.m_d_f4InterpolationLUT,
															 g_cuHandleICGN.m_d_RDescent,
															 g_cuHandleICGN.m_d_fSubsetAveR,
															 g_cuHandleICGN.m_d_fSubsetT,
															 g_cuHandleICGN.m_d_fSubsetAveT,
															 g_cuHandleICGN.m_d_iIterationNums,
															 g_cuHandleICGN.m_d_dP);
	// For Debug
	/*real_t *test1;
	hcreateptr(test1, m_iPOINumber*(m_iSubsetSize + 1));
	cudaMemcpy(test1, g_cuHandleICGN.m_d_fSubsetAveT, sizeof(real_t)*m_iPOINumber*(m_iSubsetSize + 1), cudaMemcpyDeviceToHost);
	std::cout << test1[0] << ", " << std::endl;
	hdestroyptr(test);*/

	// For debug
	/*real_t *dp;
	hcreateptr(dp, m_iPOINumber * 6);
	cudaMemcpy(dp, g_cuHandleICGN.m_d_dP, sizeof(real_t)*m_iPOINumber * 6, cudaMemcpyDeviceToHost);
	std::cout << dp[0] << ", " << dp[3] << std::endl;
	hdestroyptr(dp);*/

}

void cuICGN2D::cuFinalize()
{		
	// ICGN calculation parameters
	cudaSafeFree(g_cuHandleICGN.m_d_fRx);
	cudaSafeFree(g_cuHandleICGN.m_d_fRy);
	cudaSafeFree(g_cuHandleICGN.m_d_fTx);
	cudaSafeFree(g_cuHandleICGN.m_d_fTy);
	cudaSafeFree(g_cuHandleICGN.m_d_fTxy);
	cudaSafeFree(g_cuHandleICGN.m_d_f4InterpolationLUT);

	cudaSafeFree(g_cuHandleICGN.m_d_iIterationNums);

	cudaSafeFree(g_cuHandleICGN.m_d_fSubsetR);
	cudaSafeFree(g_cuHandleICGN.m_d_fSubsetT);
	cudaSafeFree(g_cuHandleICGN.m_d_fSubsetAveR);
	cudaSafeFree(g_cuHandleICGN.m_d_fSubsetAveT);
	cudaSafeFree(g_cuHandleICGN.m_d_invHessian);
	cudaSafeFree(g_cuHandleICGN.m_d_RDescent);
	cudaSafeFree(g_cuHandleICGN.m_d_dP);
}

}// namespace paDIC
}// namespace TW

//__global__ void ICGN_Computation_Kernel(real_t* d_fU, real_t *d_fV,
//										// Inputs
//										const int* d_dPXY,
//										const int iImgWidth, const int iImgHeight,
//										const int iStartX, const int iStartY,
//										const int iROIWidth, const int iROIHeight,
//										const int iSubsetX, const int iSubsetY,
//										const int iSubsetW, const int iSubsetH,
//										const int m_iMaxIteration,
//										const real_t fDeltaP,
//										const uchar1 *tarImg,
//										real_t* whole_d_dInvHessian,
//										real_t4 *fInterpolation,
//										real_t2 *whole_d_2dRDescent,
//										real_t *whole_d_dSubsetAveR,
//										// Tempts
//										real_t* whole_d_dSubsetT,
//										real_t* whole_d_dSubsetAveT,
//										// Outputs
//										int *whole_d_iIteration,
//										real_t *whole_d_dP)
//{
//	__shared__ real_t sm[BLOCK_SIZE_64];
//	__shared__ real_t m_dDP[6];
//	__shared__ real_t m_dWarp[6];
//	__shared__ real_t m_dP[6];
//	__shared__ int_t break_sig[1];
//
//	const int tid = threadIdx.x;
//	const int dim = blockDim.x;
//	const int bid = blockIdx.x;
//
//	real_t m_dWarpX, m_dWarpY;
//	int m_iTempX, m_iTempY;
//	real_t m_dTempX, m_dTempY;
//	real_t dtemptVal;//=0;
//	int size = iSubsetH*iSubsetW;
//
//	// Get the local handles for every CUDA block
//	real_t *m_dSubsetT = whole_d_dSubsetT + iSubsetH*iSubsetW*bid;
//	real_t *m_dSubsetAveT = whole_d_dSubsetAveT + (iSubsetH*iSubsetW + 1)*bid;
//	real_t *m_dInvHessian = whole_d_dInvHessian + bid * 36;
//	real_t2 *m_dRDescent = whole_d_2dRDescent + bid*iSubsetH*iSubsetW * 3;
//	real_t *m_dSubsetAveR = whole_d_dSubsetAveR + bid*(iSubsetH*iSubsetW + 1);
//
//	// Initialization
//	if (tid == 0)
//	{
//		// Transfer the initial guess to IC-GN algorithm
//		m_dP[0] = d_fU[bid];
//		m_dP[1] = 0;
//		m_dP[2] = 0;
//		m_dP[3] = d_fV[bid];
//		m_dP[4] = 0;
//		m_dP[5] = 0;
//
//		// Initialize the warp matrix
//		m_dWarp[0] = 1 + m_dP[1];
//		m_dWarp[1] = m_dP[2];
//		m_dWarp[2] = m_dP[0];
//		m_dWarp[3] = m_dP[4];
//		m_dWarp[4] = 1 + m_dP[5];
//		m_dWarp[5] = m_dP[3] ;
//	}
//
//	if(tid==32)
//	{
//		break_sig[0]=0;
//	}
//	__syncthreads();
//
//	// ICGN iterations started from this point
//	int m_iIteration;
//	for (m_iIteration = 0; m_iIteration < m_iMaxIteration; m_iIteration++)
//	{
//		real_t  mySum = 0;
//		for (int id = tid; id < size; id += dim)
//		{
//			int l = id / iSubsetW;
//			int m = id % iSubsetW;
//			if(l < iSubsetH && m < iSubsetW)
//			{
//				m_dWarpX = d_dPXY[2 * bid + 1] + m_dWarp[0] * (m - iSubsetX) + m_dWarp[1] * (l - iSubsetY) + m_dWarp[2];
//				m_dWarpY = d_dPXY[2 * bid + 0] + m_dWarp[3] * (m - iSubsetX) + m_dWarp[4] * (l - iSubsetY) + m_dWarp[5];
//
//				if(m_dWarpX < (real_t)iStartX) m_dWarpX = (real_t)iStartX;
//				if(m_dWarpY < (real_t)iStartY) m_dWarpY = (real_t)iStartY;
//
//				if (m_dWarpX >= real_t(iROIWidth + iStartX)) m_dWarpX = real_t(iROIWidth + iStartX - 1);
//				if (m_dWarpY >= real_t(iROIHeight + iStartY))m_dWarpY = real_t(iROIHeight + iStartY - 1);
//
//				m_iTempX = int(m_dWarpX);
//				m_iTempY = int(m_dWarpY);
//
//				m_dTempX = m_dWarpX - m_iTempX;
//				m_dTempY = m_dWarpY - m_iTempY;
//
//				if (m_dTempX <= 0.0000001 && m_dTempY <= 0.0000001)
//				{
//					dtemptVal = (real_t)tarImg[m_iTempY * iImgWidth + m_iTempX].x;
//				}
//				else
//				{
//					// unroll the for loop
//					real_t4 a1, a2, a3, a4;
//					a1 = fInterpolation[0 * iROIWidth*iROIHeight + (m_iTempY - iStartY)*iROIWidth + (m_iTempX - iStartX)];
//					a2 = fInterpolation[1 * iROIWidth*iROIHeight + (m_iTempY - iStartY)*iROIWidth + (m_iTempX - iStartX)];
//					a3 = fInterpolation[2 * iROIWidth*iROIHeight + (m_iTempY - iStartY)*iROIWidth + (m_iTempX - iStartX)];
//					a4 = fInterpolation[3 * iROIWidth*iROIHeight + (m_iTempY - iStartY)*iROIWidth + (m_iTempX - iStartX)];
//
//					dtemptVal = 
//						a1.w * pow(m_dTempY, 0) * pow(m_dTempX, 0) +
//						a1.x * pow(m_dTempY, 0) * pow(m_dTempX, 1) +
//						a1.y * pow(m_dTempY, 0) * pow(m_dTempX, 2) +
//						a1.z * pow(m_dTempY, 0) * pow(m_dTempX, 3) +
//
//						a2.w * pow(m_dTempY, 1) * pow(m_dTempX, 0) +
//						a2.x * pow(m_dTempY, 1) * pow(m_dTempX, 1) +
//						a2.y * pow(m_dTempY, 1) * pow(m_dTempX, 2) +
//						a2.z * pow(m_dTempY, 1) * pow(m_dTempX, 3) +
//
//						a3.w * pow(m_dTempY, 2) * pow(m_dTempX, 0) +
//						a3.x * pow(m_dTempY, 2) * pow(m_dTempX, 1) +
//						a3.y * pow(m_dTempY, 2) * pow(m_dTempX, 2) +
//						a3.z * pow(m_dTempY, 2) * pow(m_dTempX, 3) +
//
//						a4.w * pow(m_dTempY, 3) * pow(m_dTempX, 0) +
//						a4.x * pow(m_dTempY, 3) * pow(m_dTempX, 1) +
//						a4.y * pow(m_dTempY, 3) * pow(m_dTempX, 2) +
//						a4.z * pow(m_dTempY, 3) * pow(m_dTempX, 3);
//				}
//				m_dSubsetT[l*iSubsetW + m] = dtemptVal;
//				mySum += dtemptVal / (real_t)size;
//			}
//		}
//		
//		__syncthreads();
//		real_t avg;
//		reduceBlock<BLOCK_SIZE_64, float>(sm, mySum, tid);
//		__syncthreads();
//		avg = sm[0];
//		mySum = 0;
//		for (int id = tid; id < size; id += dim)
//		{
//			dtemptVal = m_dSubsetT[id] - avg;
//			mySum += dtemptVal*dtemptVal;
//			m_dSubsetAveT[id + 1] = dtemptVal;
//		}
//		__syncthreads();
//		reduceBlock<BLOCK_SIZE_64, float>(sm, mySum, tid);
//		__syncthreads();
//
//		if (tid == 0)
//		{
//
//			m_dSubsetAveT[0] = dtemptVal = sqrt(sm[tid]);
//			sm[tid] = m_dSubsetAveR[0] / dtemptVal;
//		}
//
//		real_t n0, n1, n2, n3, n4, n5;
//		n0 = 0; n1 = 0; n2 = 0; n3 = 0; n4 = 0; n5 = 0;
//		real_t2 rd;
//		__syncthreads();
//		real_t Nor = sm[0];
//		for (int id = tid; id < size; id += dim)
//		{
//			dtemptVal = (Nor)* m_dSubsetAveT[id + 1] - m_dSubsetAveR[id + 1];
//			rd = m_dRDescent[id];
//			n0 += (rd.x * dtemptVal);
//			n1 += (rd.y * dtemptVal);
//			rd = m_dRDescent[size + id];
//			n2 += (rd.x * dtemptVal);
//			n3 += (rd.y * dtemptVal);
//			rd = m_dRDescent[size * 2 + id];
//			n4 += (rd.x * dtemptVal);
//			n5 += (rd.y * dtemptVal);
//		}
//		
//		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n0, tid);
//		//	__syncthreads();
//		if (tid < 6)
//			n0 = sm[0];
//		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n1, tid);
//		//	__syncthreads();
//		if (tid < 6)
//			n1 = sm[0];
//		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n2, tid);
//		//	__syncthreads();
//		if (tid < 6)
//			n2 = sm[0];
//		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n3, tid);
//		//	__syncthreads();
//		if (tid < 6)
//			n3 = sm[0];
//		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n4, tid);
//		//	__syncthreads();
//		if (tid < 6)
//			n4 = sm[0];
//		reduceBlock<BLOCK_SIZE_64, real_t>(sm, n5, tid);
//		//	__syncthreads();
//		if (tid < 6)
//			n5 = sm[0];
//		if (tid < 6)
//		{
//			m_dDP[tid] =
//				m_dInvHessian[tid * 6 + 0] * n0 +
//				m_dInvHessian[tid * 6 + 1] * n1 +
//				m_dInvHessian[tid * 6 + 2] * n2 +
//				m_dInvHessian[tid * 6 + 3] * n3 +
//				m_dInvHessian[tid * 6 + 4] * n4 +
//				m_dInvHessian[tid * 6 + 5] * n5;
//		}
//		if(tid==0)
//		{
//			dtemptVal = (1 + m_dDP[1]) * (1 + m_dDP[5]) - m_dDP[2] * m_dDP[4];
//			// when it is too small just try to make it vallid
//			//if (dtemptVal == 0)//
//			//{
//			//	dtemptVal=0.00000001;
//			//}
//			//W(P) <- W(P) o W(DP)^-1
//			//0.00041051498556043
//			m_dWarp[0] = ((1 + m_dP[1]) * (1 + m_dDP[5]) - m_dP[2] * m_dDP[4]) / dtemptVal;
//			m_dWarp[1] = (m_dP[2] * (1 + m_dDP[1]) - (1 + m_dP[1]) * m_dDP[2]) / dtemptVal;
//			m_dWarp[2] = m_dP[0] + (m_dP[2] * (m_dDP[0] * m_dDP[4] - m_dDP[3] - m_dDP[3] * m_dDP[1]) - (1 + m_dP[1]) * (m_dDP[0] * m_dDP[5] + m_dDP[0] - m_dDP[2] * m_dDP[3])) / dtemptVal;
//			m_dWarp[3] = (m_dP[4] * (1 + m_dDP[5]) - (1 + m_dP[5]) * m_dDP[4]) / dtemptVal;
//			m_dWarp[4] = ((1 + m_dP[5]) * (1 + m_dDP[1]) - m_dP[4] * m_dDP[2]) / dtemptVal;
//			m_dWarp[5] = m_dP[3] + ((1 + m_dP[5]) * (m_dDP[0] * m_dDP[4] - m_dDP[3] - m_dDP[3] * m_dDP[1]) - m_dP[4] * (m_dDP[0] * m_dDP[5] + m_dDP[0] - m_dDP[2] * m_dDP[3])) / dtemptVal;
//
//			// Update DeltaP
//			m_dP[0] = m_dWarp[2];
//			m_dP[1] = m_dWarp[0] - 1;
//			m_dP[2] = m_dWarp[1];
//			m_dP[3] = m_dWarp[5];
//			m_dP[4] = m_dWarp[3];
//			m_dP[5] = m_dWarp[4] - 1;
//		
//			if (sqrt(
//				pow(m_dDP[0], 2) + 
//				pow(m_dDP[1] * iSubsetX, 2) + 
//				pow(m_dDP[2] * iSubsetY, 2) + 
//				pow(m_dDP[3], 2) +
//				pow(m_dDP[4] * iSubsetX, 2) + 
//				pow(m_dDP[5] * iSubsetY, 2)) 
//				< fDeltaP)
//			{
//				break_sig[0]=1;
//			}
//		}
//		__syncthreads();
//		if(break_sig[0]==1)
//			break;
//	}
//	if(tid==0)
//	{
//		whole_d_iIteration[bid]=m_iIteration;
//		d_fU[bid] = m_dP[0];
//		d_fV[bid] = m_dP[3];
//	}
//	if(tid<6)
//	{
//		whole_d_dP[bid*6+tid]=m_dP[tid];
//	}
//}

//__global__ void RefAllSubetsNorm_Kernel(// Inputs
//									    const uchar1 *d_refImg,
//									    const int *d_iPOIXY,
//									    const int iSubsetX, const int iSubsetY,
//									    const int iImgWidth, const int iImgHeight,
//									    //const int iStartX, const int iStartY,
//									    //const int iROIWidth, const int iROIHeight,
//									    // Outputs
//									    real_t *d_wholeSubset,
//									    real_t *d_wholeSubsetNorm)
//{
//	__shared__ real_t sm[BLOCK_SIZE_64];
//	const int tid = threadIdx.x;
//	const int dim = blockDim.x;
//	const int bid = blockIdx.x;
//	const int iSubsetW = iSubsetX * 2 + 1;
//	const int iSubsetH = iSubsetY * 2 + 1;
//	const int iSubsetSize = iSubsetW * iSubsetH;
//	real_t avg = 0;// = 0;
//	real_t mySum = 0;
//	real_t tempt = 0;
//	real_t *dSubSet = d_wholeSubset + iSubsetSize*bid;
//	real_t *dSubsetAve = d_wholeSubsetNorm + (iSubsetSize + 1)*bid;
//
//	// Construct refImgR and compute the mean value R_m
//	for (int id = tid; id < iSubsetSize; id += dim)
//	{
//		int l = id / iSubsetW;	// y
//		int m = id % iSubsetW;	// x
//
//		tempt = (real_t)d_refImg[(d_iPOIXY[bid * 2 + 0] - iSubsetY + l)*iImgWidth + d_iPOIXY[bid * 2 + 1] - iSubsetX + m].x;
//		dSubSet[id] = tempt;
//		mySum += tempt / real_t(iSubsetSize);
//	}
//	__syncthreads();	
//	reduceBlock<BLOCK_SIZE_64, real_t>(sm, mySum, tid);
//	__syncthreads();
//
//	// Calculate Sigma_i sqrt(R_i - R_m) 
//	avg = sm[0];	// Norm
//	mySum = 0; 
//	for (int id = tid; id < iSubsetSize; id += dim)
//	{
//		tempt = dSubSet[id] - avg;	
//		mySum += tempt * tempt;
//		dSubsetAve[id + 1] = tempt;
//	}
//	__syncthreads();
//	reduceBlock<BLOCK_SIZE_64, real_t>(sm, mySum, tid);
//	__syncthreads();
//
//	if(tid ==0)
//	{
//		dSubsetAve[0] = sqrt(sm[tid]);
//	}
//}

//__global__ void InverseHessian_Kernel(// Inputs
//									  const real_t* d_Rx, 
//									  const real_t *d_Ry,
//									  const int_t *d_iPOIXY,
//									  const int_t iSubsetX, const  int_t iSubsetY,
//									  const int_t iStartX, const int_t iStartY,
//									  const int_t iROIWidth, const int_t iROIHeight,
//									  // Outputs
//									  real_t2 *whole_d_RDescent,
//									  real_t* whole_d_InvHessian)
//{
//	__shared__ real_t Hessian[96];
//	__shared__ real_t sm[BLOCK_SIZE_64];
//	__shared__ int iIndOfRowTempt[8];
//
//	const int tid = threadIdx.x;
//	const int dim = blockDim.x;
//	const int bid = blockIdx.x;
//
//	const int iSubsetW = iSubsetX * 2 + 1;
//	const int iSubsetH = iSubsetY * 2 + 1;
//	const int iSubsetSize = iSubsetW * iSubsetH;
//
//
//	real_t tempt;
//	real_t t_dD0, t_dD1, t_dD2, t_dD3, t_dD4, t_dD5;
//
//	real_t2 *RDescent = whole_d_RDescent + bid * iSubsetSize * 3;
//	real_t *r_InvHessian = whole_d_InvHessian + bid * 36;
//
//	// Initialize the Hessian Matrix to 0;
//	for (int id = tid; id < 96; id += dim)
//	{
//		Hessian[id] = 0;
//	}
//
//	// Construct the Hessian Matrix 
//	// H = Sigma_i [ \partial(W) / \partial(p) Gradient R]^T [ \partial(W) / \partial(p) Gradient R]
//	for (int id = tid; id < iSubsetSize; id += dim)
//	{
//		int l = id / iSubsetW;	// y
//		int m = id % iSubsetW;	// x
//		
//		real_t tx = d_Rx[(d_iPOIXY[bid * 2 + 0] - iSubsetY + l - iStartY)*iROIWidth + d_iPOIXY[bid * 2 + 1] - iSubsetX + m - iStartX];
//		RDescent[l*iSubsetW + m].x = t_dD0 = tx;
//		RDescent[l*iSubsetW + m].y = t_dD1 = tx * (m - iSubsetX);
//		RDescent[iSubsetSize + l * iSubsetW + m].x = t_dD2 = tx * (l - iSubsetY);
//
//		real_t ty = d_Ry[(d_iPOIXY[bid * 2 + 0] - iSubsetY + l - iStartY)*iROIWidth + d_iPOIXY[bid * 2 + 1] - iSubsetX + m - iStartX];
//		RDescent[iSubsetSize + l * iSubsetW + m].y = t_dD3 = ty;
//		RDescent[iSubsetSize * 2 + l * iSubsetW + m].x = t_dD4 = ty * (m - iSubsetX);
//		RDescent[iSubsetSize * 2 + l * iSubsetW + m].y = t_dD5 = ty * (l - iSubsetY);
//
//		//00		
//		tempt = t_dD0 * t_dD0;
//		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
//		if (tid == 0)
//		{
//			Hessian[0 * 16 + 0] += sm[0];
//		}
////11
//		tempt = t_dD1 * t_dD1;
//		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
//		if (tid == 0)
//		{
//			Hessian[1 * 16 + 1] += sm[0];
//		}
////22		
//		tempt = t_dD2 * t_dD2;
//		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
//		if (tid == 0)
//		{
//			Hessian[2 * 16 + 2] += sm[0];
//		}
////33
//		tempt = t_dD3 * t_dD3;
//		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
//		if (tid == 0)
//		{
//			Hessian[3 * 16 + 3] += sm[0];
//		}
////44		
//		tempt=t_dD4 * t_dD4; 
//		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
//		if(tid==0)
//		{
//			Hessian[4*16+4]+=sm[0];
//		}
////55
//		tempt=t_dD5 * t_dD5; 
//		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
//		if(tid==0)
//		{
//			Hessian[5*16+5]+=sm[0];
//		}
//
//
////01		
//		tempt=t_dD0 * t_dD1; 
//		reduceBlock<BLOCK_SIZE_64, real_t>(sm, tempt, tid);
//		if(tid==0)
//		{
//			Hessian[0*16+1]+=sm[0];
//		}
////02
//		tempt=t_dD0 * t_dD2; 
//		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
//		if(tid==0)
//		{
//			Hessian[0*16+2]+=sm[0];
//		}
////03		
//		tempt=t_dD0 * t_dD3; 
//		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
//		if(tid==0)
//		{
//			Hessian[0*16+3]+=sm[0];
//		}
////04
//		tempt=t_dD0 * t_dD4; 
//		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
//		if(tid==0)
//		{
//			Hessian[0*16+4]+=sm[0];
//		}
////05		
//		tempt=t_dD0 * t_dD5; 
//		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
//		if(tid==0)
//		{
//			Hessian[0*16+5]+=sm[0];
//		}
////12
//		tempt=t_dD1 * t_dD2; 
//		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
//		if(tid==0)
//		{
//			Hessian[1*16+2]+=sm[0];
//		}
////13		
//		tempt=t_dD1 * t_dD3; 
//		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
//		if(tid==0)
//		{
//			Hessian[1*16+3]+=sm[0];
//		}
////14
//		tempt=t_dD1 * t_dD4; 
//		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
//		if(tid==0)
//		{
//			Hessian[1*16+4]+=sm[0];
//		}
////15		
//		tempt=t_dD1 * t_dD5; 
//		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
//		if(tid==0)
//		{
//			Hessian[1*16+5]+=sm[0];
//		}
//
//
//
////23		
//		tempt=t_dD2 * t_dD3; 
//		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
//		if(tid==0)
//		{
//			Hessian[2*16+3]+=sm[0];
//		}
////24
//		tempt=t_dD2 * t_dD4; 
//		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
//		if(tid==0)
//		{
//			Hessian[2*16+4]+=sm[0];
//		}
////25		
//		tempt=t_dD2 * t_dD5; 
//		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
//		if(tid==0)
//		{
//			Hessian[2*16+5]+=sm[0];
//		}
//
//
////34
//		tempt=t_dD3 * t_dD4; 
//		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
//		if(tid==0)
//		{
//			Hessian[3*16+4]+=sm[0];
//		}
////35		
//		tempt=t_dD3 * t_dD5; 
//		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
//		if(tid==0)
//		{
//			Hessian[3*16+5]+=sm[0];
//		}
//
////45		
//		tempt=t_dD4 * t_dD5; 
//		reduceBlock<BLOCK_SIZE_64,float>(sm,tempt,tid);
//		if(tid==0)
//		{
//			Hessian[4*16+5]+=sm[0];
//		}
//		__syncthreads();
//		if (tid < BLOCK_SIZE_64)
//			sm[tid] = 0;
//	}
//
//	if(tid<5)
//	{
//		Hessian[(tid + 1) * 16 + 0] = Hessian[0 * 16 + (tid + 1)];
//	}
//	if(tid<4)
//	{
//		Hessian[(tid + 2) * 16 + 1] = Hessian[1 * 16 + (tid + 2)];
//	}
//	if(tid<3)
//	{
//		Hessian[(tid + 3) * 16 + 2] = Hessian[2 * 16 + (tid + 3)];
//	}
//	if(tid<2)
//	{
//		Hessian[(tid + 4) * 16 + 3] = Hessian[3 * 16 + (tid + 4)];
//	}
//	if(tid==0)
//	{
//		Hessian[5 * 16 + 4] = Hessian[4 * 16 + 5];
//	}
//
//	// Initialize the Hessian matrix 
//	if(tid<6)
//	{
//		Hessian[tid * 16 + tid + 8] = 1;
//	}
//
//	// Pivoting
//	if (tid < 16)
//	{
//		for (int l = 0; l < 6; l++)
//		{
//			//Find pivot (maximum lth column element) in the rest (6-l) rows
//			//找到最大值
//			if (tid < 8)
//			{
//				iIndOfRowTempt[tid] = l;
//			}
//			if (tid < 6 - l)
//			{
//				iIndOfRowTempt[tid] = tid + l;
//			}
//			if (tid < 4)
//			{
//				if (Hessian[iIndOfRowTempt[tid] * 16 + l] < Hessian[iIndOfRowTempt[tid + 4] * 16 + l])
//					iIndOfRowTempt[tid] = iIndOfRowTempt[tid + 4];
//			}
//			if (tid < 2)
//			{
//				if (Hessian[iIndOfRowTempt[tid] * 16 + l] < Hessian[iIndOfRowTempt[tid + 2] * 16 + l])
//					iIndOfRowTempt[tid] = iIndOfRowTempt[tid + 2];
//			}
//			if (tid == 0)
//			{
//				if (Hessian[iIndOfRowTempt[tid] * 16 + l] < Hessian[iIndOfRowTempt[tid + 1] * 16 + l])
//					iIndOfRowTempt[tid] = iIndOfRowTempt[tid + 1];
//				
//				// Maximum's ind is stored in iIndOfRowTempt[0]
//				if (Hessian[iIndOfRowTempt[tid] * 16 + l] < 0.0000001)
//				{
//					Hessian[iIndOfRowTempt[tid] * 16 + l] = 0.0000001;
//				}
//			}
//			if (tid < 12)
//			{
//				int m_iIndexOfCol = tid / 6 * 8 + tid % 6;
//				float m_dTempt;
//				//swap 操作
//				if (iIndOfRowTempt[0] != l)
//				{
//					m_dTempt = Hessian[iIndOfRowTempt[0] * 16 + m_iIndexOfCol];
//					Hessian[iIndOfRowTempt[0] * 16 + m_iIndexOfCol] = Hessian[l * 16 + m_iIndexOfCol];
//					Hessian[l * 16 + m_iIndexOfCol] = m_dTempt;
//				}
//
//
//				// Perform row operation to form required identity matrix out of the Hessian matrix
//				Hessian[l * 16 + m_iIndexOfCol] /= Hessian[l * 16 + l];
//
//				for (int next_row = 0; next_row < 6; next_row++)
//				{
//					if (next_row != l)
//					{
//						Hessian[next_row * 16 + m_iIndexOfCol] -= Hessian[l * 16 + m_iIndexOfCol] * Hessian[next_row * 16 + l];
//					}
//				}
//			}
//		}
//	}
//
//
//	//inv Hessian
//	if (tid < 32)
//		r_InvHessian[tid] = Hessian[tid / 6 * 16 + tid % 6 + 8];
//	if (tid < 4)
//	{
//		//	tid+=32;
//		r_InvHessian[tid + 32] = Hessian[(tid + 32) / 6 * 16 + (tid + 32) % 6 + 8];
//	}
//}