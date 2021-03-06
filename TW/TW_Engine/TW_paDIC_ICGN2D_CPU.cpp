#include "TW_paDIC_ICGN2D_CPU.h"
#include "TW_MemManager.h"
#include "TW_utils.h"
#include "TW_StopWatch.h"

#include <QDebug>
#include <mkl.h>
#include <omp.h>

namespace TW{
namespace paDIC{

ICGN2D_CPU::ICGN2D_CPU(
	int_t iImgWidth, int_t iImgHeight,
	int_t iStartX, int_t iStartY,
	int_t iROIWidth, int_t iROIHeight,
	int_t iSubsetX, int_t iSubsetY,
	int_t iNumberX, int_t iNumberY,
	int_t iNumIterations,
	real_t fDeltaP,
	ICGN2DInterpolationFLag Iflag,
	paDICThreadFlag Tflag)
	: ICGN2D(iImgWidth, iImgHeight,
		 	 iStartX, iStartY,
			 iROIWidth, iROIHeight,
			 iSubsetX, iSubsetY,
			 iNumberX, iNumberY,
			 iNumIterations,
			 fDeltaP)
	, m_Iflag(Iflag)
	, m_Tflag(Tflag)
{
	ICGN2D_Prepare();	
}

ICGN2D_CPU::ICGN2D_CPU(
	const cv::Mat& refImg,
	int_t iImgWidth, int_t iImgHeight,
	int_t iStartX, int_t iStartY,
	int_t iROIWidth, int_t iROIHeight,
	int_t iSubsetX, int_t iSubsetY,
	int_t iNumberX, int_t iNumberY,
	int_t iNumIterations,
	real_t fDeltaP,
	ICGN2DInterpolationFLag Iflag,
	paDICThreadFlag Tflag)
	: ICGN2D(refImg,
			 iImgWidth, iImgHeight,
		 	 iStartX, iStartY,
			 iROIWidth, iROIHeight,
			 iSubsetX, iSubsetY,
			 iNumberX, iNumberY,
			 iNumIterations,
			 fDeltaP)
	, m_Iflag(Iflag)
	, m_Tflag(Tflag)
{
	ICGN2D_Prepare();	
}

ICGN2D_CPU::~ICGN2D_CPU()
{
	ICGN2D_Finalize();
}

void ICGN2D_CPU::ResetRefImg(const cv::Mat& refImg)
{
	m_refImg = refImg;
	m_isRefImgUpdated = true;
}

void ICGN2D_CPU::SetTarImg(const cv::Mat& tarImg)
{
	m_tarImg = tarImg;
}

void ICGN2D_CPU::ICGN2D_Algorithm(
	real_t *fU,
	real_t *fV,
	int *iNumIterations,
	int *iPOIpos,
	const cv::Mat& tarImg)
{
	m_tarImg = tarImg;
	
	double start = omp_get_wtime();
	ICGN2D_Precomputation();
	double end = omp_get_wtime();

	std::cout << "Time for Precomputation is: " << 1000 * (end - start) << std::endl;

	start = omp_get_wtime();
	switch (m_Tflag)
	{
	case TW::paDIC::paDICThreadFlag::Single:
	{
		for (int i = 0; i < m_iPOINumber; i++)
		{
			ICGN2D_Compute(fU[i],
						   fV[i],
						   iNumIterations[i],
						   iPOIpos[2 * i + 1],
						   iPOIpos[2 * i + 0],
						   i);
		}
		break;
	}

	case TW::paDIC::paDICThreadFlag::Multicore:
	{
#pragma	omp parallel for
		for (int i = 0; i < m_iPOINumber; i++)
		{
			ICGN2D_Compute(fU[i],
						   fV[i],
						   iNumIterations[i],
						   iPOIpos[2 * i + 1],
						   iPOIpos[2 * i + 0],
						   i);
		}
		break;
	}

	default:
	{
		break;
	}
	}
	end = omp_get_wtime();
	std::cout << "ICGN time is: " << 1000 * (end - start) << " [ms]" << std::endl;
	//std::cout << fU[0] << ", " << fV[0] << ", " << std::endl;
}

void ICGN2D_CPU::ICGN2D_Algorithm(
	real_t *fU,
	real_t *fV,
	int *iNumIterations,
	int *iPOIpos,
	const cv::Mat& tarImg,
	float &fPrecomputeTime,
	float &fICGNTime)
{
	m_tarImg = tarImg;
	
	double start = omp_get_wtime();
	ICGN2D_Precomputation();
	double end = omp_get_wtime();

	fPrecomputeTime = 1000 * (end - start);

	//std::cout << "Time for Precomputation is: " << 1000 * (end - start) << std::endl;

	start = omp_get_wtime();
	switch (m_Tflag)
	{
	case TW::paDIC::paDICThreadFlag::Single:
	{
		for (int i = 0; i < m_iPOINumber; i++)
		{
			ICGN2D_Compute(fU[i],
						   fV[i],
						   iNumIterations[i],
						   iPOIpos[2 * i + 1],
						   iPOIpos[2 * i + 0],
						   i);
		}
		break;
	}

	case TW::paDIC::paDICThreadFlag::Multicore:
	{
#pragma	omp parallel for
		for (int i = 0; i < m_iPOINumber; i++)
		{
			ICGN2D_Compute(fU[i],
						   fV[i],
						   iNumIterations[i],
						   iPOIpos[2 * i + 1],
						   iPOIpos[2 * i + 0],
						   i);
		}
		break;
	}

	default:
	{
		break;
	}
	}
	end = omp_get_wtime();
	
	fICGNTime = 1000 * (end - start);
	// std::cout << "ICGN time is: " << 1000 * (end - start) << " [ms]" << std::endl;
	//std::cout << fU[0] << ", " << fV[0] << ", " << std::endl;
}


void ICGN2D_CPU::ICGN2D_Algorithm(
	real_t *fU,
	real_t *fV,
	int *iNumIterations,
	int *iPOIpos,
	const cv::Mat& tarImg,
	int_t head, 
	int_t tail)
{
	m_tarImg = tarImg;
	
	double start = omp_get_wtime();
	ICGN2D_Precomputation();
	double end = omp_get_wtime();

	std::cout << "Time for Precomputation is: " << 1000 * (end - start) << std::endl;

	start = omp_get_wtime();
	switch (m_Tflag)
	{
	case TW::paDIC::paDICThreadFlag::Single:
	{
		for (int i = 0; i < m_iPOINumber; i++)
		{
			ICGN2D_Compute(fU[i],
						   fV[i],
						   iNumIterations[i],
						   iPOIpos[2 * i + 1],
						   iPOIpos[2 * i + 0],
						   i);
		}
		break;
	}

	case TW::paDIC::paDICThreadFlag::Multicore:
	{
#pragma	omp parallel for
		for (int i = 0; i < m_iPOINumber; i++)
		{
			ICGN2D_Compute(fU[i],
						   fV[i],
						   iNumIterations[i],
						   iPOIpos[2 * i + 1],
						   iPOIpos[2 * i + 0],
						   i);
		}
		break;
	}

	default:
	{
		break;
	}
	}
	end = omp_get_wtime();
	std::cout << "ICGN time is: " << 1000 * (end - start) << " [ms]" << std::endl;
}


void ICGN2D_CPU::ICGN2D_Precomputation_Prepare()
{
	switch (m_Iflag)
	{
	case TW::paDIC::ICGN2DInterpolationFLag::Bicubic:
	{
		hcreateptr<real_t>(m_fRx, m_iROIHeight, m_iROIWidth);
		hcreateptr<real_t>(m_fRy, m_iROIHeight, m_iROIWidth);
		hcreateptr<real_t>(m_fTx, m_iROIHeight, m_iROIWidth);
		hcreateptr<real_t>(m_fTy, m_iROIHeight, m_iROIWidth);
		hcreateptr<real_t>(m_fTxy, m_iROIHeight, m_iROIWidth);
		hcreateptr<real_t>(m_fInterpolation, m_iROIHeight, m_iROIWidth, 4, 4);
		break;
	}

	case TW::paDIC::ICGN2DInterpolationFLag::BicubicSpline:
	{
		hcreateptr<real_t>(m_fRx, m_iROIHeight, m_iROIWidth);
		hcreateptr<real_t>(m_fRy, m_iROIHeight, m_iROIWidth);
		hcreateptr<real_t>(m_fInterpolation, m_iROIHeight, m_iROIWidth, 4, 4);
		break;
	}

	default:
	{
		break;
	}
	}
}

void ICGN2D_CPU::ICGN2D_Precomputation() 
{
	
	switch (m_Tflag)
	{
	case TW::paDIC::paDICThreadFlag::Single:
	{
		switch (m_Iflag)
		{
		case TW::paDIC::ICGN2DInterpolationFLag::Bicubic:
		{
			// Compute gradients of m_refImg & m_tarImg
			GradientXY_2Images_s(m_refImg,
								 m_tarImg,
								 m_iStartX, m_iStartY,
								 m_iROIWidth, m_iROIHeight,
								 m_refImg.cols, m_refImg.rows,
								 TW::AccuracyOrder::Quadratic,
								 m_fRx,
								 m_fRy,
								 m_fTx,
								 m_fTy,
								 m_fTxy);
			// Compute the LUT for bicubic interpolation
			BicubicCoefficients_s(m_tarImg,
								  m_fTx,
								  m_fTy,
								  m_fTxy,
								  m_iStartX, m_iStartY,
								  m_iROIWidth, m_iROIHeight,
								  m_tarImg.cols, m_tarImg.cols,
								  m_fInterpolation);
			break;
		}

		case TW::paDIC::ICGN2DInterpolationFLag::BicubicSpline:
		{
			// Compute gradients of m_refImg
			Gradient_s(m_refImg,
				  	   m_iStartX, m_iStartY,
					   m_iROIWidth, m_iROIHeight,
					   m_refImg.cols, m_refImg.rows,
					   TW::AccuracyOrder::Quadratic,
					   m_fRx,
					   m_fRy);
			// Compute the LUT for bicubic B-Spline interpolation
			BicubicSplineCoefficients_s(m_tarImg,
										m_iStartX,	
										m_iStartY,
										m_iROIWidth,
										m_iROIHeight,
										m_tarImg.cols,
										m_tarImg.rows,
										m_fInterpolation);
			break;
		}

		default:
		{
			break;
		}
		
		}
		break;
	}

	case TW::paDIC::paDICThreadFlag::Multicore:
	{
		switch (m_Iflag)
		{
		case TW::paDIC::ICGN2DInterpolationFLag::Bicubic:
		{
			// Compute gradients of m_refImg & m_tarImg
			GradientXY_2Images_m(m_refImg,
								 m_tarImg,
								 m_iStartX, m_iStartY,
								 m_iROIWidth, m_iROIHeight,
								 m_refImg.cols, m_refImg.rows,
								 TW::AccuracyOrder::Quadratic,
								 m_fRx,
								 m_fRy,
								 m_fTx,
								 m_fTy,
								 m_fTxy);
			// Compute the LUT for bicubic interpolation
			BicubicCoefficients_m(m_tarImg,
								  m_fTx,
								  m_fTy,
								  m_fTxy,
								  m_iStartX, m_iStartY,
								  m_iROIWidth, m_iROIHeight,
								  m_tarImg.cols, m_tarImg.cols,
								  m_fInterpolation);

			break;
		}

		case TW::paDIC::ICGN2DInterpolationFLag::BicubicSpline:
		{
			// Compute gradients of m_refImg
			Gradient_m(m_refImg,
				  	   m_iStartX, m_iStartY,
					   m_iROIWidth, m_iROIHeight,
					   m_refImg.cols, m_refImg.rows,
					   TW::AccuracyOrder::Quadratic,
					   m_fRx,
					   m_fRy);
			// Compute the LUT for bicubic B-Spline interpolation
			BicubicSplineCoefficients_m(m_tarImg,
										m_iStartX,	
										m_iStartY,
										m_iROIWidth,
										m_iROIHeight,
										m_tarImg.cols,
										m_tarImg.rows,
										m_fInterpolation);

			break;
		}

		default:
		{
			break;
		}
		}
		break;
	}

	default:
	{
		break;
	}
	}

	// For debug
	/*std::cout<<"Bicubic First: "<<std::endl;
	std::cout.precision(10);
	for(int i=0;i<4;i++)
	{
		for(int j=0;j<4;j++)
		{
			std::cout<<m_fInterpolation[0][0][i][j]<<", ";
		}
		std::cout<<std::endl;
	}

	for(int i=0;i<4;i++)
	{
		for(int j=0;j<4;j++)
		{
			std::cout<<m_fInterpolation[m_iROIHeight-1][m_iROIWidth-1][i][j]<<", ";
		}
		std::cout<<"\n";
	}*/
}

void ICGN2D_CPU::ICGN2D_Precomputation_Finalize()
{

	switch (m_Iflag)	
	{
	case TW::paDIC::ICGN2DInterpolationFLag::Bicubic:
	{
		hdestroyptr(m_fRx);
		hdestroyptr(m_fRy);
		hdestroyptr(m_fTx);
		hdestroyptr(m_fTy);
		hdestroyptr(m_fTxy);
		hdestroyptr(m_fInterpolation);
	}
		break;

	case TW::paDIC::ICGN2DInterpolationFLag::BicubicSpline:
	{
		hdestroyptr(m_fRx);
		hdestroyptr(m_fRy);
		hdestroyptr(m_fInterpolation);
	}
		break;
	default:
		break;
	}	
}

void ICGN2D_CPU::ICGN2D_Prepare()
{
	// Allocate required memory
	ICGN2D_Precomputation_Prepare();

	hcreateptr(m_fSubsetR, m_iPOINumber, m_iSubsetH, m_iSubsetW);
	hcreateptr(m_fSubsetT, m_iPOINumber, m_iSubsetH, m_iSubsetW);
	hcreateptr(m_fRDescent,m_iPOINumber, m_iSubsetH, m_iSubsetW, 6);
}

ICGN2DFlag ICGN2D_CPU::ICGN2D_Compute(real_t &fU,
								  	  real_t &fV,
									  int &iNumIterations,
									  const int iPOIx,
								      const int iPOIy,
									  const int id)
{
	// Eqn.(1)
	// H(delta(p)) = RHS =
	//		Sigma_i Sigma_j{
	//				[gradient(R)(\partial(W)/\partial(p)]^T [R_ij - R_m] }
	//	   -R_s/T_s Sigma_i Sigma_j{
	//				[gradient(R)(\partial(W)/\partial(p)]^T  [T_m - T_ij] }
	// where R_s = sqrt[ Sigma_ij(R_ij - R_m)^2], T_s = sqrt[ Sigma_ij(T_ij - T_m)^2]

	real_t fError = 0;										// R_s / T_s ( T_i - R_i)
	real_t fRefSubsetMean = 0, fTarSubsetMean = 0;			// Mean intensity values within Ref & Tar subsets
	real_t fRefSubsetNorm = 0, fTarSubsetNorm = 0;			// sqrt(Sigma(R_i - R_mean)^2) Normalization parameter for Ref & Tar subsets
	std::vector<real_t> v_P(6,0);							// Deformation parameter P(u,ux,uy,v,vx,vy);
	std::vector<real_t> v_dP(6,0);							// delta(p) in Eqn.(1): Incremental P, dP(du, dux, duy, dv, dvx, dvy);
	std::vector<real_t> v_RHS(6, 0);						// The RHS vector of Eqn.(1)
	// matrix \partial(W) / \partial(p) 2-by-6				
	std::vector<std::vector<real_t>> m_Jacobian(2, std::vector<real_t>(6, 0));
	// The warp matrix W = 
	//					|1+u_x		u_y		u|
	//					|  v_x	  1+v_y		v|
	//					|	 0		  0		1|
	std::vector<std::vector<real_t>> m_W(3, std::vector<real_t>(3, 0));
	// The Hessian Matrix H = [gradient(R)(\partial(W)/\partial(p)]^T * [gradient(R)(\partial(W)/\partial(p)]
	std::vector<real_t> m_Hessian(6*6, 0);

	// Only when the reference image is changed does these happen
	//if(m_isRefImgUpdated)
	//{	
	// Precompute all the invariant paramters before the iterations
	for (int l = 0; l < m_iSubsetH; l++)
	{
		for (int m = 0; m < m_iSubsetW; m++)
		{
			// x and y indices of each pixel in the subset  
			int idY = iPOIy - m_iSubsetY + l;
			int idX = iPOIx - m_iSubsetX + m;

			// Construct the Ref subset
			m_fSubsetR[id][l][m] =
				static_cast<real_t>(m_refImg.at<uchar>(idY, idX));

			// Compute the Sigma_i Sigma_j(R_ij)
			fRefSubsetMean += m_fSubsetR[id][l][m];

			// Calculate the Jacobian:  \partial(W) / \partial(p)
			// | 1 dx dy  0  0  0 |, dx = m - m_iSubsetX, distance to the POI
			// | 0  0  0  1 dx dy |	 dy = l - m_iSubsetY
			m_Jacobian[0][0] = 1;	m_Jacobian[0][1] = real_t(m - m_iSubsetX);	m_Jacobian[0][2] = real_t(l - m_iSubsetY);	m_Jacobian[0][3] = 0;	m_Jacobian[0][4] = 0;						m_Jacobian[0][5] = 0;
			m_Jacobian[1][0] = 0;	m_Jacobian[1][1] = 0;						m_Jacobian[1][2] = 0;						m_Jacobian[1][3] = 1;	m_Jacobian[1][4] = real_t(m - m_iSubsetX);	m_Jacobian[1][5] = real_t(l - m_iSubsetY);

			// Calculate gradient(R)(\partial(W)/\partial(p)
			// | Rx Ry | * | 1 dx dy  0  0  0 |
			//			   | 0  0  0  1 dx dy |
			for (int k = 0; k < 6; k++)
			{
				m_fRDescent[id][l][m][k] =
					m_fRx[idY - m_iStartY][idX - m_iStartX] * m_Jacobian[0][k] +
					m_fRy[idY - m_iStartY][idX - m_iStartX] * m_Jacobian[1][k];
			}

			// Calculate Hessian Matrix H = 
			//							Sigma_i Sigma_j (m_fRDescent^T * m_fRDescent)
			// Note: Since H is symmetric, only calculate its lowever-triangle elements
			//		| H_00	   0     0	   0	 0	   0 |
			//		| H_10	H_11	 0	   0	 0	   0 |
			// H =	| H_20	H_21  H_22	   0	 0	   0 |
			//		| H_30  H_31  H_32  H_33	 0	   0 |
			//		| H_40  H_41  H_42  H_43  H_44	   0 |
			//		| H_50  H_51  H_52  H_53  H_54  H_55 |
			for (int k = 0; k < 6; k++)
			{
				for (int n = 0; n <= k; n++)
				{
					m_Hessian[k * 6 + n] += m_fRDescent[id][l][m][k] * m_fRDescent[id][l][m][n];
				}
			}
		}
	}
	//}
	// 
	
	//// For debug use
	/*std::cout << "Hessian Before\n";
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			std::cout << m_Hessian[i*6+j] << ",\t";
		}
		std::cout << "\n";
	}*/

	// Calculate R_m and make sure R_m != 0
	fRefSubsetMean /= real_t(m_iSubsetSize);	// R_m
	if(std::abs(fRefSubsetMean) <= std::numeric_limits<real_t>::epsilon())
		return ICGN2DFlag::DarkSubset;

	// Calculate R_s = sqrt[ Sigma_ij(R_ij - R_m)^2] and make sure R_s != 0;
	for(int l = 0; l < m_iSubsetH; l++)
	{
		for(int m = 0; m < m_iSubsetW; m++)
		{
			m_fSubsetR[id][l][m] = m_fSubsetR[id][l][m] - fRefSubsetMean;	// R_ij - R_m
			fRefSubsetNorm += m_fSubsetR[id][l][m] * m_fSubsetR[id][l][m];	// Sigma_ij (R_ij - R_m)^2;
		}
	}
	fRefSubsetNorm = std::sqrt(fRefSubsetNorm);	// R_s
	if(std::abs(fRefSubsetNorm) <= std::numeric_limits<real_t>::epsilon())
		return ICGN2DFlag::DarkSubset;

	// Initialize deformation vector P and its incremental one dP
	// v_P = {u, 0, 0, v, 0, 0}, v_dP = {0, 0, 0, 0, 0, 0}
	//       (u,ux,uy, v,vx,vy}
	//		 {0, 1, 2, 3, 4, 5}
	v_P[0] = fU;
	v_P[3] = fV;

	// Initialize the warp matrix W
	m_W[0][0] = 1 + v_P[1];		m_W[0][1] = v_P[2];			m_W[0][2] = v_P[0];
	m_W[1][0] = v_P[4];			m_W[1][1] = 1 + v_P[5];		m_W[1][2] = v_P[3];
	m_W[2][0] = 0;				m_W[2][1] = 0;				m_W[2][2] = 1;

	// Construct the warpped subset T: T_ij
	for(int l=0; l < m_iSubsetH; l++)
	{
		for(int m = 0; m < m_iSubsetW; m++)
		{
			// Calculate the subpixel location within the subset T
			// |WarpX|	 | POIx| |1+u_x		u_y		u| | m - iSubsetX|
			// |WarpY| = | POIy|+|  v_x	  1+v_y		v|*| l - iSubsetY|
			// |   1 |	 |    1| | 0		  0		1| |			1|
			real_t fWarpX = iPOIx + m_W[0][0] * (m - m_iSubsetX) + m_W[0][1] * (l - m_iSubsetY) + m_W[0][2];
			real_t fWarpY = iPOIy + m_W[1][0] * (m - m_iSubsetX) + m_W[1][1] * (l - m_iSubsetY) + m_W[1][2];
			int iIntPixX = int(fWarpX);
			int iIntPixY = int(fWarpY);

			// Make sure that iIntPixX & iIntPixY are within the 
			// ROI[iStartX~iStartX+iROIWidth-1][iStartY~iStartY+iROIHeight-1]
			if( (iIntPixX >= m_iStartX) && (iIntPixY >= m_iStartY) && 
				(iIntPixX < m_iStartX + m_iROIWidth) && (iIntPixY < m_iStartY + m_iROIHeight))	
			{
				// Initially this is the interger locations
				m_fSubsetT[id][l][m] = static_cast<real_t>(m_tarImg.at<uchar>(iIntPixY,iIntPixX));
				fTarSubsetMean += m_fSubsetT[id][l][m];
			}
			else
			{
				return ICGN2DFlag::OutofROI;
			}
		}
	}
	fTarSubsetMean /= real_t(m_iSubsetSize);
	if(std::abs(fTarSubsetMean) <= std::numeric_limits<real_t>::epsilon())
		return ICGN2DFlag::DarkSubset;

	// Calculate T_s = sqrt[ Sigma_ij(T_ij - T_m)^2] and make sure T_s != 0;
	for(int l = 0; l < m_iSubsetH; l++)
	{
		for(int m = 0; m < m_iSubsetW; m++)
		{
			m_fSubsetT[id][l][m] = m_fSubsetT[id][l][m] - fTarSubsetMean;	// T_i - T_m	
			fTarSubsetNorm += m_fSubsetT[id][l][m] * m_fSubsetT[id][l][m];	// sigma (T_i - T_m)^2
		}
	}
	fTarSubsetNorm = sqrt(fTarSubsetNorm);	// sqrt(Sigma(T_i - T_m)^2
	if(std::abs(fTarSubsetNorm) <= std::numeric_limits<real_t>::epsilon())
		return ICGN2DFlag::DarkSubset;

	// Construct the RHS vector v_RHS
	//real_t fError = 0;	// R_s / T_s ( T_i - R_i)
	for(int l = 0; l < m_iSubsetH; l++)
	{
		for(int m = 0; m < m_iSubsetW; m++)
		{
			fError = (fRefSubsetNorm / fTarSubsetNorm) * m_fSubsetT[id][l][m] - m_fSubsetR[id][l][m]; 

			// Calculate the RHS = Sigma{[m_RDescent]^T[ R_s / T_s  * T_i - R_i]}
			for(int k = 0; k < 6; k++)
				v_RHS[k] += (m_fRDescent[id][l][m][k] * fError);
		}
	}

	// For Debug
	/*for(int i=0; i< v_RHS.size(); i++)
		std::cout<<v_RHS[i]<<", ";
	std::cout<<std::endl;*/

	// Using MKL's LAPACK routing to solve the linear equations ""m_H * v_dP = v_RHS""
	// H is symmetricúČ but not guaranteed to be positive definite
	MKL_INT ipiv[6];
#ifdef TW_USE_DOUBLE
	int infor = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', 6, m_Hessian.data(), 6);
	LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', 6, 1, m_Hessian.data(), 6, v_RHS.data(), 1);
	//int infor = LAPACKE_dsysv(LAPACK_ROW_MAJOR, 'L', 6, 1, m_Hessian.data(), 6, ipiv, v_RHS.data(), 1);;
#else
	int infor = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', 6, m_Hessian.data(), 6);
	LAPACKE_spotrs(LAPACK_ROW_MAJOR, 'L', 6, 1, m_Hessian.data(), 6, v_RHS.data(), 1);
	//int infor = LAPACKE_ssysv(LAPACK_ROW_MAJOR, 'L', 6, 1, m_Hessian.data(), 6, ipiv, v_RHS.data(), 1);
#endif // TW_USE_DOUBLE
	
	// Check for the exact singularity 
	if (infor > 0) {
		qDebug() << "The element of the diagonal factor ";
		qDebug() << "D(" << infor << "," << infor << ") is zero, so that D is singular;\n";
		qDebug() << "the solution could not be computed.\n";
		return ICGN2DFlag::SingularHessian;
	}
	//// For Debug
	/*std::cout << "\nHessian After\n";
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			std::cout << m_Hessian[i*6+j] << ",\t";
		}
		std::cout << "\n";
	}
	std::cout<<"DeltaP Befor"<<std::endl;
	for(int i=0; i< v_RHS.size(); i++)
		std::cout<<v_RHS[i]<<", ";

	std::cout<<std::endl;*/

	// NOTE: Now dP's value is stored in v_RHS [ du dux duy dv dvx dvy]
	// Update warp m_W and deformation parameter p v_P
	// W(P) <- W(P) o W(DP)^-1
	real_t fTemp = (1 + v_RHS[1]) * (1 + v_RHS[5]) - v_RHS[2] * v_RHS[4];
	if(std::abs(fTemp) <= std::numeric_limits<real_t>::epsilon())
		return ICGN2DFlag::SingularWarp;

	// Update m_W
	m_W[0][0] = ((1 + v_P[1]) * (1 + v_RHS[5]) - v_P[2] * v_RHS[4]) / fTemp;
	m_W[0][1] = (v_P[2] * (1 + v_RHS[1]) - (1 + v_P[1]) * v_RHS[2]) / fTemp;
	m_W[0][2] = v_P[0] + (v_P[2] * (v_RHS[0] * v_RHS[4] - v_RHS[3] - v_RHS[3] * v_RHS[1]) - (1 + v_P[1]) * (v_RHS[0] * v_RHS[5] + v_RHS[0] - v_RHS[2] * v_RHS[3])) / fTemp;
	m_W[1][0] = (v_P[4] * (1 + v_RHS[5]) - (1 + v_P[5]) * v_RHS[4]) / fTemp;
	m_W[1][1] = ((1 + v_P[5]) * (1 + v_RHS[1]) - v_P[4] * v_RHS[2]) / fTemp;
	m_W[1][2] = v_P[3] + ((1 + v_P[5]) * (v_RHS[0] * v_RHS[4] - v_RHS[3] - v_RHS[3] * v_RHS[1]) - v_P[4] * (v_RHS[0] * v_RHS[5] + v_RHS[0] - v_RHS[2] * v_RHS[3])) / fTemp;
	m_W[2][0] = 0;
	m_W[2][1] = 0;
	m_W[2][2] = 1;

	// For Debug
	/*std::cout<<"Warp Befor"<<std::endl;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			std::cout << m_W[i][j]<< ",\t";
		}
		std::cout << "\n";
	}*/

	// Update P & the output fU&fV
	v_P[0] = fU = m_W[0][2];
	v_P[1] = m_W[0][0] - 1;
	v_P[2] = m_W[0][1];
	v_P[3] = fV = m_W[1][2];
	v_P[4] = m_W[1][0];
	v_P[5] = m_W[1][1] - 1;


	/* Perform the ICGN iterative optimizatin from this point, with preset maximum iteration step*/
	iNumIterations = 1;
	while (iNumIterations < m_iNumIterations && 
		   sqrt(pow(v_RHS[0], 2) + pow(v_RHS[1] * m_iSubsetX, 2) + pow(v_RHS[2] * m_iSubsetY, 2) + pow(v_RHS[3], 2) + pow(v_RHS[4] * m_iSubsetX, 2) + pow(v_RHS[5] * m_iSubsetY, 2)) >= m_fDeltaP)
	{
		++iNumIterations;

		// Fill the warpped subset T (sub-pixel positions)
		fTarSubsetMean = fTarSubsetNorm = 0;
		for(int l=0; l < m_iSubsetH; l++)
		{
			for (int m = 0; m < m_iSubsetW; m++)
			{
				real_t fWarpX = iPOIx + m_W[0][0] * (m - m_iSubsetX) + m_W[0][1] * (l - m_iSubsetY) + m_W[0][2];
				real_t fWarpY = iPOIy + m_W[1][0] * (m - m_iSubsetX) + m_W[1][1] * (l - m_iSubsetY) + m_W[1][2];
				int iIntPixX = int(fWarpX);
				int iIntPixY = int(fWarpY);

				// Make sure that iIntPixX & iIntPixY are within the 
				// ROI[iStartX~iStartX+iROIWidth-1][iStartY~iStartY+iROIHeight-1]
				if ((iIntPixX >= m_iStartX) && (iIntPixY >= m_iStartY) &&
					(iIntPixX < m_iStartX + m_iROIWidth) && (iIntPixY < m_iStartY + m_iROIHeight))
				{
					real_t fTempX = fWarpX - real_t(iIntPixX);
					real_t fTempY = fWarpY - real_t(iIntPixY);

					// Mostly, T contains sub-pixel locations. These locations can be estimated by interpolation
					m_fSubsetT[id][l][m] = 0;
					for (int k = 0; k < 4; k++)
					{
						for (int n = 0; n < 4; n++)
						{
							// Note the indices in m_fBsplineInterpolation should minus the start position [X,Y]
							// of the ROI
							m_fSubsetT[id][l][m] += m_fInterpolation[iIntPixY - m_iStartY][iIntPixX - m_iStartX][k][n] * pow(fTempY, k) * pow(fTempX, n);
						}
					}
					fTarSubsetMean += m_fSubsetT[id][l][m];
				}
				else
				{
					return ICGN2DFlag::OutofROI;
				}
			}
		}
		fTarSubsetMean /= real_t(m_iSubsetSize);	// T_m
		if(std::abs(fTarSubsetMean) <= std::numeric_limits<real_t>::epsilon())
			return ICGN2DFlag::DarkSubset;

		// Calculate sqrt(Sigma(T_i - T_m)^2)
		for (int l = 0; l < m_iSubsetH; l++)
		{
			for (int m = 0; m < m_iSubsetW; m++)
			{
				m_fSubsetT[id][l][m] = m_fSubsetT[id][l][m] - fTarSubsetMean;	// T_i - T_m
				fTarSubsetNorm += m_fSubsetT[id][l][m] * m_fSubsetT[id][l][m];	// Sigma(T_i - T_m)^2
			}
		}
		fTarSubsetNorm = sqrt(fTarSubsetNorm);	// sqrt(Sigma(T_i - T_m)^2)
		if(std::abs(fTarSubsetNorm) <= std::numeric_limits<real_t>::epsilon())
			return ICGN2DFlag::DarkSubset;

		// Construct the RHS vector v_RHS
		for (int k = 0; k < 6; k++)
			v_RHS[k] = 0;

		
		for (int l = 0; l < m_iSubsetH; l++)
		{
			for (int m = 0; m < m_iSubsetW; m++)
			{
				fError  = (fRefSubsetNorm / fTarSubsetNorm) * m_fSubsetT[id][l][m];
				fError -= m_fSubsetR[id][l][m];
				
				// Calculate the RHS = Sigma{[m_RDescent]^T[ R_s / T_s  * T_i - R_i]}
				for (int k = 0; k < 6; k++)
					v_RHS[k] += (m_fRDescent[id][l][m][k] * fError);
			}
		}

		// For Debug
		
		/*if (iNumIterations == 2)
		{
			std::cout<<"Third Iteration: "<<"\n";
			std::cout.precision(10);
			std::cout<<m_fSubsetT[id][0][0]<<std::endl;
			std::cout<<m_fSubsetR[id][32][32]<<std::endl;
			std::cout << fRefSubsetNorm <<", "<<", "<<fTarSubsetMean<<", "<<fTarSubsetNorm<<", "<<fError
				<<", "<<m_fSubsetT[id][m_iSubsetH-1][m_iSubsetW-1]<<","<<m_fSubsetR[id][m_iSubsetH-1][m_iSubsetW-1]<<", "
				<< (fRefSubsetNorm / fTarSubsetNorm)<<", "
				<< (fRefSubsetNorm / fTarSubsetNorm)* m_fSubsetT[id][m_iSubsetH-1][m_iSubsetW-1]<<", "
				<< (fRefSubsetNorm / fTarSubsetNorm)* m_fSubsetT[id][m_iSubsetH-1][m_iSubsetW-1] - m_fSubsetR[id][m_iSubsetH-1][m_iSubsetW-1]<<"\n";

			for (int i = 0; i < v_RHS.size(); i++)
				std::cout << v_RHS[i] << ", ";
		}*/

		// Using MKL's LAPACK routing to solve the linear equations ""m_H * v_dP = v_RHS""
		// H is symmetricúČ but not guaranteed to be positive definite
#ifdef TW_USE_DOUBLE
		infor = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', 6, 1, m_Hessian.data(), 6, v_RHS.data(), 1);
		//int infor = LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N', 6, 1, m_Hessian.data(), 6, v_RHS.data(), 1);
		/*int infor = LAPACKE_dsysv(LAPACK_ROW_MAJOR, 'L', 6, 1, m_Hessian.data(), 6, ipiv, v_RHS.data(), 1);;*/
#else
		infor = LAPACKE_spotrs(LAPACK_ROW_MAJOR, 'L', 6, 1, m_Hessian.data(), 6, v_RHS.data(), 1);
		//int infor = LAPACKE_ssysv(LAPACK_ROW_MAJOR, 'L', 6, 1, m_Hessian.data(), 6, ipiv, v_RHS.data(), 1);
#endif // TW_USE_DOUBLE

		/*if (iNumIterations == 2)
		{
			std::cout<<"\nDeltaP after for Iteration 2"<<"\n";
			for (int i = 0; i < v_RHS.size(); i++)
				std::cout << v_RHS[i] << ", ";
		}*/

		// Check for the exact singularity 
		if (infor < 0) {
			qDebug() << "The element of the diagonal factor ";
			qDebug() << "D(" << infor << "," << infor << ") is ellegal, so that\n";
			qDebug() << "the solution could not be computed.\n";
			return ICGN2DFlag::SingularHessian;
		}

		// NOTE: Now dP's value is stored in v_RHS [ du dux duy dv dvx dvy]
		// Update warp m_W and deformation parameter p v_P
		// W(P) <- W(P) o W(DP)^-1
		real_t fTemp = (1 + v_RHS[1]) * (1 + v_RHS[5]) - v_RHS[2] * v_RHS[4];
		if (std::abs(fTemp) <= std::numeric_limits<real_t>::epsilon())
			return ICGN2DFlag::SingularWarp;

		// Update m_W
		m_W[0][0] = ((1 + v_P[1]) * (1 + v_RHS[5]) - v_P[2] * v_RHS[4]) / fTemp;
		m_W[0][1] = (v_P[2] * (1 + v_RHS[1]) - (1 + v_P[1]) * v_RHS[2]) / fTemp;
		m_W[0][2] = v_P[0] + (v_P[2] * (v_RHS[0] * v_RHS[4] - v_RHS[3] - v_RHS[3] * v_RHS[1]) - (1 + v_P[1]) * (v_RHS[0] * v_RHS[5] + v_RHS[0] - v_RHS[2] * v_RHS[3])) / fTemp;
		m_W[1][0] = (v_P[4] * (1 + v_RHS[5]) - (1 + v_P[5]) * v_RHS[4]) / fTemp;
		m_W[1][1] = ((1 + v_P[5]) * (1 + v_RHS[1]) - v_P[4] * v_RHS[2]) / fTemp;
		m_W[1][2] = v_P[3] + ((1 + v_P[5]) * (v_RHS[0] * v_RHS[4] - v_RHS[3] - v_RHS[3] * v_RHS[1]) - v_P[4] * (v_RHS[0] * v_RHS[5] + v_RHS[0] - v_RHS[2] * v_RHS[3])) / fTemp;
		m_W[2][0] = 0;
		m_W[2][1] = 0;
		m_W[2][2] = 1;

		// For Debug
		/*std::cout<<"Warp After"<<std::endl;
		for (int i = 0; i < 3; i++)
		{
		for (int j = 0; j < 3; j++)
		{
		std::cout << m_W[i][j]<< ",\t";
		}
		std::cout << "\n";
		}*/

		// Update P & the output fU&fV
		v_P[0] = fU = m_W[0][2];
		v_P[1] = m_W[0][0] - 1;
		v_P[2] = m_W[0][1];
		v_P[3] = fV = m_W[1][2];
		v_P[4] = m_W[1][0];
		v_P[5] = m_W[1][1] - 1;
	}

	// For Debug
	/*std::cout<<fTarSubsetMean<<std::endl;
	std::cout<<fTarSubsetNorm<<std::endl;
	std::cout<<iNumIterations<<std::endl;
	std::cout<<"Warp After"<<std::endl;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			std::cout << m_W[i][j]<< ",\t";
		}
		std::cout << "\n";
	}*/

	return ICGN2DFlag::Success;
}

void ICGN2D_CPU::ICGN2D_Finalize()
{
	ICGN2D_Precomputation_Finalize();

	hdestroyptr(m_fSubsetR);
	hdestroyptr(m_fSubsetT);
	hdestroyptr(m_fRDescent);
}

} //!- namespace paDIC
} //!- namespace TW