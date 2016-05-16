
#include "TW_paDIC_cuFFTCC2D.h"
#include "TW_utils.h"
#include "TW_MemManager.h"
#include "TW_paDIC_ICGN2D_CPU.h"
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <gtest\gtest.h>

using namespace TW;

//TEST(Gradient, Gradient_s)
//{
//	cv::Mat mat = cv::imread("Example2\\crop_oht_cfrp_00.bmp");
//
//	auto imgWidth = mat.cols;
//	auto imgHeight= mat.rows;
//
//	cv::Mat matS(cv::Size(imgWidth, imgHeight), CV_8UC1);
//	cv::cvtColor(mat, matS, CV_BGR2GRAY);
//
//	float **Gx, **Gy/*, *Gxy*/;
//
//	hcreateptr(Gx, (imgHeight-2), (imgWidth-2));
//	hcreateptr(Gy, (imgHeight-2), (imgWidth-2));
//	//hcreateptr(Gxy, (imgWidth-2)*(imgHeight-2));
//
//	try{
//		Gradient_s(matS, 1,1,imgWidth-2, imgHeight-2, imgWidth, imgHeight, TW::AccuracyOrder::Quadratic, Gx, Gy/*, Gxy*/);
//	}
//	catch(const char* c)
//	{
//		std::cerr<<c<<std::endl;
//	}
//	std::cout<<(float)matS.at<uchar>(0,1)<<", "<<(float)matS.at<uchar>(0,2)<<", "<<(float)matS.at<uchar>(0,3)<<std::endl;
//	std::cout<<(float)matS.at<uchar>(1,1)<<", "<<(float)matS.at<uchar>(1,2)<<", "<<(float)matS.at<uchar>(1,3)<<std::endl;
//	std::cout<<(float)matS.at<uchar>(2,1)<<", "<<(float)matS.at<uchar>(2,2)<<", "<<(float)matS.at<uchar>(2,3)<<std::endl;
//	std::cout<<"GradientX: "<<Gx[0][1]<<", "<<"GradientY: "<<Gy[0][1]<<std::endl;
//
//	hdestroyptr(Gx);
//	hdestroyptr(Gy);
//	//hdestroyptr(Gxy);
//}
//
//TEST(GradientGPU, Gradient_P)
//{
//	cv::Mat mat = cv::imread("Example2\\crop_oht_cfrp_00.bmp");
//	cv::Mat mat1= cv::imread("Example2\\crop_oht_cfrp_01.bmp");
//
//	auto imgWidth = mat.cols;
//	auto imgHeight= mat.rows;
//
//	cv::Mat matS(cv::Size(imgWidth, imgHeight), CV_8UC1);
//	cv::Mat matS1(cv::Size(imgWidth, imgHeight), CV_8UC1);
//	cv::cvtColor(mat, matS, CV_BGR2GRAY);
//	cv::cvtColor(mat1, matS1, CV_BGR2GRAY);
//
//	float **Fx, **Fy;
//	float **Gx, **Gy ,**Gxy;
//
//	hcreateptr(Fx, (imgHeight-2), (imgWidth-2));
//	hcreateptr(Fy, (imgHeight-2), (imgWidth-2));
//	hcreateptr(Gx, (imgHeight-2), (imgWidth-2));
//	hcreateptr(Gy, (imgHeight-2), (imgWidth-2));
//	hcreateptr(Gxy, (imgWidth-2)*(imgHeight-2));
//
//	try{
//		Gradient_s(matS, 1,1,imgWidth-2, imgHeight-2, imgWidth, imgHeight, TW::AccuracyOrder::Quadratic, Gx, Gy/*, Gxy*/);
//	}
//	catch(const char* c)
//	{
//		std::cerr<<c<<std::endl;
//	}
//	std::cout<<(float)matS.at<uchar>(0,1)<<", "<<(float)matS.at<uchar>(0,2)<<", "<<(float)matS.at<uchar>(0,3)<<std::endl;
//	std::cout<<(float)matS.at<uchar>(1,1)<<", "<<(float)matS.at<uchar>(1,2)<<", "<<(float)matS.at<uchar>(1,3)<<std::endl;
//	std::cout<<(float)matS.at<uchar>(2,1)<<", "<<(float)matS.at<uchar>(2,2)<<", "<<(float)matS.at<uchar>(2,3)<<std::endl;
//	std::cout<<"GradientX: "<<Gx[0][1]<<", "<<"GradientY: "<<Gy[0][1]<<std::endl;
//	std::cout<<"GradientX: "<<Gx[3][3]<<", "<<"GradientY: "<<Gy[3][3]<<std::endl;
//
//	 GPU rountines
//	uchar1 *imgF, *imgG;
//	cudaMalloc((void**)&imgF, imgHeight*imgWidth);
//	cudaMalloc((void**)&imgG, imgHeight*imgWidth);
//
//	cudaMemcpy(imgF,  (void*)matS.data, imgHeight*imgWidth,cudaMemcpyHostToDevice);
//	cudaMemcpy(imgG,  (void*)matS1.data,imgHeight*imgWidth,cudaMemcpyHostToDevice);
//
//	float *dFx, *dFy, *dGx, *dGy, *dGxy;
//	cudaMalloc((void**)&dFx, sizeof(float)*(imgHeight-2)*(imgWidth-2));
//	cudaMalloc((void**)&dFy, sizeof(float)*(imgHeight-2)*(imgWidth-2));
//	cudaMalloc((void**)&dGx, sizeof(float)*(imgHeight-2)*(imgWidth-2));
//	cudaMalloc((void**)&dGy, sizeof(float)*(imgHeight-2)*(imgWidth-2));
//	cudaMalloc((void**)&dGxy, sizeof(float)*(imgHeight - 2)*(imgWidth - 2));
//
//	cuGradientXY_2Images(imgF,imgG,1,1,imgWidth-2,imgHeight-2,imgWidth,imgHeight, TW::AccuracyOrder::Quadratic, dFx,dFy,dGx,dGy,dGxy);
//	cudaMemcpy(Fx[0], dFx, sizeof(float)*(imgHeight-2)*(imgWidth-2),cudaMemcpyDeviceToHost);
//	cudaMemcpy(Fy[0], dFy, sizeof(float)*(imgHeight-2)*(imgWidth-2),cudaMemcpyDeviceToHost);
//	std::cout<<"GradientX: "<<Fx[0][1]<<", "<<"GradientY: "<<Fy[0][1]<<std::endl;
//	std::cout<<"GradientX: "<<Fx[imgHeight-3][imgWidth-3]<<", "<<"GradientY: "<<Fy[imgHeight-3][imgWidth-3]<<std::endl;
//	
//	cuGradient(imgF,3,3,imgWidth-6,imgHeight-6,imgWidth,imgHeight, TW::AccuracyOrder::Quadratic, dFx,dFy);
//	cudaMemcpy(Fx[0], dFx, sizeof(float)*(imgHeight-2)*(imgWidth-2),cudaMemcpyDeviceToHost);
//	cudaMemcpy(Fy[0], dFy, sizeof(float)*(imgHeight-2)*(imgWidth-2),cudaMemcpyDeviceToHost);
//	std::cout<<"GradientX: "<<Fx[0][0]<<", "<<"GradientY: "<<Fy[0][0]<<std::endl;
//	std::cout<<"GradientX: "<<Fx[imgHeight-3][imgWidth-3]<<", "<<"GradientY: "<<Fy[imgHeight-3][imgWidth-3]<<std::endl;
//
//
//	cudaFree(dFx);
//	cudaFree(dFy);
//	cudaFree(dGx);
//	cudaFree(dGy);
//	cudaFree(dGxy);
//	cudaFree(imgF);
//	cudaFree(imgG);
//
//	hdestroyptr(Fx);
//	hdestroyptr(Fy);
//	hdestroyptr(Gx);
//	hdestroyptr(Gy);
//	hdestroyptr(Gxy);
//
//
//
//
//
//
//
//	hcreateptr(Fx, (imgHeight-6), (imgWidth-6));
//	hcreateptr(Fy, (imgHeight-6), (imgWidth-6));
//	hcreateptr(Gx, (imgHeight-6), (imgWidth-6));
//	hcreateptr(Gy, (imgHeight-6), (imgWidth-6));
//	hcreateptr(Gxy, (imgWidth-6)*(imgHeight-6));
//
//	try{
//		Gradient_s(matS, 3,3,imgWidth-6, imgHeight-6, imgWidth, imgHeight, TW::AccuracyOrder::Quadratic, Gx, Gy/*, Gxy*/);
//	}
//	catch(const char* c)
//	{
//		std::cerr<<c<<std::endl;
//	}
//	std::cout<<(float)matS.at<uchar>(0,1)<<", "<<(float)matS.at<uchar>(0,2)<<", "<<(float)matS.at<uchar>(0,3)<<std::endl;
//	std::cout<<(float)matS.at<uchar>(1,1)<<", "<<(float)matS.at<uchar>(1,2)<<", "<<(float)matS.at<uchar>(1,3)<<std::endl;
//	std::cout<<(float)matS.at<uchar>(2,1)<<", "<<(float)matS.at<uchar>(2,2)<<", "<<(float)matS.at<uchar>(2,3)<<std::endl;
//	std::cout<<"GradientX: "<<Gx[0][1]<<", "<<"GradientY: "<<Gy[0][1]<<std::endl;
//	std::cout<<"GradientX: "<<Gx[3][3]<<", "<<"GradientY: "<<Gy[3][3]<<std::endl;
//
//	 GPU rountines
//
//	cudaMalloc((void**)&imgF, imgHeight*imgWidth);
//	cudaMalloc((void**)&imgG, imgHeight*imgWidth);
//
//	cudaMemcpy(imgF,  (void*)matS.data, imgHeight*imgWidth,cudaMemcpyHostToDevice);
//	cudaMemcpy(imgG,  (void*)matS1.data,imgHeight*imgWidth,cudaMemcpyHostToDevice);
//
//
//	cudaMalloc((void**)&dFx, sizeof(float)*(imgHeight-6)*(imgWidth-6));
//	cudaMalloc((void**)&dFy, sizeof(float)*(imgHeight-6)*(imgWidth-6));
//	cudaMalloc((void**)&dGx, sizeof(float)*(imgHeight-6)*(imgWidth-6));
//	cudaMalloc((void**)&dGy, sizeof(float)*(imgHeight-6)*(imgWidth-6));
//	cudaMalloc((void**)&dGxy, sizeof(float)*(imgHeight - 6)*(imgWidth - 6));
//
//	cuGradientXY_2Images(imgF,imgG,3,3,imgWidth-6,imgHeight-6,imgWidth,imgHeight, TW::AccuracyOrder::Quadratic, dFx,dFy,dGx,dGy,dGxy);
//	cudaMemcpy(Fx[0], dFx, sizeof(float)*(imgHeight-6)*(imgWidth-6),cudaMemcpyDeviceToHost);
//	cudaMemcpy(Fy[0], dFy, sizeof(float)*(imgHeight-6)*(imgWidth-6),cudaMemcpyDeviceToHost);
//	std::cout<<"GradientX: "<<Fx[0][1]<<", "<<"GradientY: "<<Fy[0][1]<<std::endl;
//	std::cout<<"GradientX: "<<Fx[3][3]<<", "<<"GradientY: "<<Fy[3][3]<<std::endl;
//	
//	cuGradient(imgF,3,3,imgWidth-6,imgHeight-6,imgWidth,imgHeight, TW::AccuracyOrder::Quadratic, dFx,dFy);
//	cudaMemcpy(Fx[0], dFx, sizeof(float)*(imgHeight-6)*(imgWidth-6),cudaMemcpyDeviceToHost);
//	cudaMemcpy(Fy[0], dFy, sizeof(float)*(imgHeight-6)*(imgWidth-6),cudaMemcpyDeviceToHost);
//	std::cout<<"GradientX: "<<Fx[0][1]<<", "<<"GradientY: "<<Fy[0][1]<<std::endl;
//	std::cout<<"GradientX: "<<Fx[3][3]<<", "<<"GradientY: "<<Fy[3][3]<<std::endl;
//	std::cout<<"GradientX: "<<Fx[imgHeight-3][imgWidth-3]<<", "<<"GradientY: "<<Fy[imgHeight-3][imgWidth-3]<<std::endl;
//
//
//	cudaFree(dFx);
//	cudaFree(dFy);
//	cudaFree(dGx);
//	cudaFree(dGy);
//	cudaFree(dGxy);
//	cudaFree(imgF);
//	cudaFree(imgG);
//
//	hdestroyptr(Fx);
//	hdestroyptr(Fy);
//	hdestroyptr(Gx);
//	hdestroyptr(Gy);
//	hdestroyptr(Gxy);
//
//}



//
//TEST(Bicubic, BicubicInterpolation)
//{
//	cv::Mat mat = cv::imread("Example2\\crop_oht_cfrp_01.bmp");
//
//	auto imgWidth = mat.cols;
//	auto imgHeight= mat.rows;
//
//	auto iROIWidth = imgWidth - 4;
//	auto iROIHeight = imgHeight - 4;
//
//	cv::Mat matS(cv::Size(imgWidth, imgHeight), CV_8UC1);
//	cv::cvtColor(mat, matS, CV_BGR2GRAY);
//
//	float****fBSpline;
//
//	float **Tx,**Ty, **Txy;
//
//	// CPU
//	hcreateptr(Tx,  iROIHeight, iROIWidth);
//	hcreateptr(Ty,  iROIHeight, iROIWidth);
//	hcreateptr(Txy, iROIHeight, iROIWidth);
//	hcreateptr(fBSpline, iROIHeight, iROIWidth, 4, 4);
//	GradientXY_s(matS,2, 2, iROIWidth, iROIHeight, imgWidth, imgHeight,TW::AccuracyOrder::Quadratic,Tx,Ty,Txy);
//	BicubicCoefficients_s(matS, Tx, Ty, Txy, 2, 2, iROIWidth, iROIHeight, imgWidth, imgHeight, fBSpline);
//
//
//	std::cout<<"First: "<<std::endl;
//	for(int i=0;i<4;i++)
//	{
//		for(int j=0;j<4;j++)
//		{
//			std::cout<<fBSpline[0][0][i][j]<<", ";
//		}
//		std::cout<<std::endl;
//	}
//	hdestroyptr(fBSpline);
//	hdestroyptr(Tx);
//	hdestroyptr(Ty);
//	hdestroyptr(Txy);
//
//	// GPU
//	uchar1 *dImgT;
//	cudaMalloc((void**)&dImgT, imgHeight*imgWidth);
//	cudaMemcpy(dImgT, (void*)matS.data, imgHeight*imgWidth, cudaMemcpyHostToDevice);
//
//	float4* dBicubic, *hBicubic;
//	cudaMalloc((void**)&dBicubic, sizeof(float4)*4*iROIHeight*iROIWidth);
//	hcreateptr<float4>(hBicubic, 4*iROIHeight*iROIWidth);
//
//	float *dFx, *dFy, *dGx, *dGy, *dGxy;
//	cudaMalloc((void**)&dFx, sizeof(float)*iROIHeight*iROIWidth);
//	cudaMalloc((void**)&dFy, sizeof(float)*iROIHeight*iROIWidth);
//	cudaMalloc((void**)&dGx, sizeof(float)*iROIHeight*iROIWidth);
//	cudaMalloc((void**)&dGy, sizeof(float)*iROIHeight*iROIWidth);
//	cudaMalloc((void**)&dGxy, sizeof(float)*iROIHeight*iROIWidth);
//
//	cuGradientXY_2Images(dImgT, dImgT, 2, 2, iROIWidth, iROIHeight, imgWidth, imgHeight, TW::AccuracyOrder::Quadratic, dFx, dFy, dGx, dGy, dGxy);
//	cuBicubicCoefficients(dImgT, dGx, dGy, dGxy, 2, 2, iROIWidth, iROIHeight, imgWidth, imgHeight, dBicubic);
//
//	cudaMemcpy(hBicubic, dBicubic, sizeof(float4)*iROIHeight*iROIWidth*4, cudaMemcpyDeviceToHost);
//
//	std::cout<<"First: "<<std::endl;
//	for(int i=0;i<4;i++)
//	{
//		
//		{
//			std::cout<<hBicubic[i*iROIHeight*iROIWidth].w<<", "<<hBicubic[i*iROIHeight*iROIWidth].x<<", "
//				<<hBicubic[i*iROIHeight*iROIWidth].y<<", "<<hBicubic[i*iROIHeight*iROIWidth].z<<", ";
//		}
//		std::cout<<std::endl;
//	}
//
//	hdestroyptr(hBicubic);
//	cudaFree(dImgT);
//	cudaFree(dBicubic);
//	cudaFree(dFx);
//	cudaFree(dFy);
//	cudaFree(dGx);
//	cudaFree(dGy);
//	cudaFree(dGxy);
//}

//TEST(BSpline, BSplineInterpolation)
//{                                                  
//	cv::Mat mat = cv::imread("Example2\\crop_oht_cfrp_01.bmp");
//
//	auto imgWidth = mat.cols;
//	auto imgHeight= mat.rows;
//
//	cv::Mat matS(cv::Size(imgWidth, imgHeight), CV_8UC1);
//	cv::cvtColor(mat, matS, CV_BGR2GRAY);
//
//	float****fBSpline;
//
//	hcreateptr(fBSpline, imgHeight-4, imgWidth-4, 4, 4);
//	BicubicSplineCoefficients_s(matS, 2, 2, imgWidth-4, imgHeight-4, imgWidth, imgHeight, fBSpline);
//
//
//	std::cout<<"First: "<<std::endl;
//	for(int i=0;i<4;i++)
//	{
//		for(int j=0;j<4;j++)
//		{
//			std::cout<<fBSpline[0][0][i][j]<<", ";
//		}
//		std::cout<<std::endl;
//	}
//	hdestroyptr(fBSpline);
//}
//
TEST(ICGN2D, ICGN2D_CPU_Hessian)
{
	cv::Mat mat = cv::imread("Example1\\fu_0.bmp");
	cv::Mat mat1= cv::imread("Example1\\fu_1.bmp");

	auto imgWidth = mat.cols;
	auto imgHeight= mat.rows;

	cv::Mat matR(cv::Size(imgWidth, imgHeight), CV_8UC1);
	cv::Mat matT(cv::Size(imgWidth, imgHeight), CV_8UC1);
	cv::cvtColor(mat, matR, CV_BGR2GRAY);
	cv::cvtColor(mat1, matT, CV_BGR2GRAY);


	TW::paDIC::ICGN2D_CPU icgn(matR,matT,
								2,2,
								imgWidth-4,imgHeight-4,
								16,16,
								92,92,
								20,
								0.001,
								TW::paDIC::ICGN2DInterpolationFLag::BicubicSpline,
								TW::paDIC::ICGN2DThreadFlag::Single);

	icgn.ICGN2D_Prepare();
	float u=0, v=0;
	int iter = 0;
	icgn.ICGN2D_Compute(u,v, iter, 28,28,0);

	std::cout<<"The displacement is [" << u << ", " << v << "]\n";
	std::cout<<iter<<"\n";

	icgn.ICGN2D_Finalize();
}

//TEST(ICGN2D, ICGN2D_CPU_All_Subsets)
//{
//	cv::Mat mat = cv::imread("Example1\\fu_0.bmp");
//	cv::Mat mat1= cv::imread("Example1\\fu_1.bmp");
//
//	auto imgWidth = mat.cols;
//	auto imgHeight= mat.rows;
//	auto m_iROIWidth = mat.cols - 4;
//	auto m_iROIHeight = mat.rows - 4;
//	
//	int m_iSubsetX = 16;
//	int m_iSubsetY = 16;
//	int	m_iMarginX = 10;
//	int m_iMarginY = 10;
//	int m_iGridSpaceX = 5;
//	int m_iGridSpaceY = 5;
//
//	cv::Mat matR(cv::Size(imgWidth, imgHeight), CV_8UC1);
//	cv::Mat matT(cv::Size(imgWidth, imgHeight), CV_8UC1);
//	cv::cvtColor(mat, matR, CV_BGR2GRAY);
//	cv::cvtColor(mat1, matT, CV_BGR2GRAY);
//
//	int_t m_iNumPOIX = int_t(floor((m_iROIWidth - m_iSubsetX * 2 - m_iMarginX * 2) / real_t(m_iGridSpaceX))) + 1;
//	int_t m_iNumPOIY = int_t(floor((m_iROIHeight - m_iSubsetY * 2 - m_iMarginY * 2) / real_t(m_iGridSpaceY))) + 1;
//
//	float *fU, *fV;
//	int *iters;
//	hcreateptr(fU, m_iNumPOIX * m_iNumPOIY);
//	hcreateptr(fV, m_iNumPOIX * m_iNumPOIY);
//	hcreateptr(iters, m_iNumPOIX * m_iNumPOIY);
//
//	int *hPOI, *dPOI;
//	hcreateptr(hPOI, m_iNumPOIX*m_iNumPOIY);
//	cudaMalloc((void**)&dPOI, sizeof(int)*m_iNumPOIX*m_iNumPOIY);
//
//	cuComputePOIPositions(dPOI, hPOI, 2, 2,
//		m_iNumPOIX, m_iNumPOIY,m_iMarginX, m_iMarginY, m_iSubsetX, m_iSubsetY, m_iGridSpaceX, m_iGridSpaceY);
//	
//
//	TW::paDIC::ICGN2D_CPU icgn(matR,matT,
//								2,2,
//								m_iROIWidth, m_iROIHeight,
//								m_iSubsetX, m_iSubsetY,
//								m_iNumPOIX, m_iNumPOIY,
//								20,
//								0.001f,
//								TW::paDIC::ICGN2DInterpolationFLag::BicubicSpline,
//								TW::paDIC::ICGN2DThreadFlag::Single);
//
//	icgn.ICGN2D_Prepare();
//	
//	icgn.ICGN2D_Algorithm(fU, fV, iters, hPOI);
//
//	std::cout<<"POI number is: "<<m_iNumPOIX * m_iNumPOIY<<"\n";
//	
//	for(int i = 0; i<m_iNumPOIY; i++)
//	{
//		for(int j=0; j<m_iNumPOIX; j++)
//		{
//			std::cout<<hPOI[(i*m_iNumPOIX+j)*2+1]<<", "<<hPOI[(i*m_iNumPOIX+j)*2+0]<<", "
//				<<fU[i*m_iNumPOIX+j]<<", "<<fV[i*m_iNumPOIX+j]<<", "<<iters[i*m_iNumPOIX+j]<<"\n";
//		}
//	}
//
//	cudaFree(dPOI);
//	hdestroyptr(fU);
//	hdestroyptr(fV);
//	hdestroyptr(iters);
//
//	icgn.ICGN2D_Finalize();
//}