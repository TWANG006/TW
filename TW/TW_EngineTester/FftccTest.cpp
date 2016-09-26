#include "TW_paDIC_cuFFTCC2D.h"
#include "TW_paDIC_FFTCC2D_CPU.h"
#include "TW_utils.h"
#include "TW_MemManager.h"
#include "TW_StopWatch.h"
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <omp.h>
#include <gtest\gtest.h>
#include <fstream>
//
using namespace TW;
//
TEST(Fftcc2D, Fftcc2D_CPU_part)
{
	cv::Mat Rmat = cv::imread("Example1\\crop_oht_cfrp_01.jpg");
	cv::Mat Tmat = cv::imread("Example1\\crop_oht_cfrp_02.jpg");

	auto wm_iWidth = Rmat.cols;
	auto wm_iHeight = Rmat.rows;

	cv::Mat Rmatnew(cv::Size(wm_iWidth, wm_iHeight), CV_8UC1);
	cv::Mat Tmatnew(cv::Size(wm_iWidth, wm_iHeight), CV_8UC1);

	cv::cvtColor(Rmat, Rmatnew, CV_BGR2GRAY);
	cv::cvtColor(Tmat, Tmatnew, CV_BGR2GRAY);

	int*** iPOIXY;
	float **fU, **fV, **fZNCC;

	paDIC::Fftcc2D_CPU *wfcc = new paDIC::Fftcc2D_CPU(
		Rmatnew.cols, Rmatnew.rows,
		50, 50,
		Rmatnew.cols-2*50, Rmatnew.rows-2*50,
		15, 15,
		40, 40,
		10, 10,
		TW::paDIC::paDICThreadFlag::Single);

	wfcc->InitializeFFTCC(
		Rmatnew,
		iPOIXY,
		fU,
		fV,
		fZNCC);

	omp_set_num_threads(12);

	StopWatch w;
	w.start();
	wfcc->Algorithm_FFTCC(
		Tmatnew,
		iPOIXY,
		fU,
		fV,
		fZNCC);
	w.stop();


	std::ofstream file;
	file.open("result.txt", std::ios::out | std::ios::trunc);
	for(int i=0; i<wfcc->GetNumPOIsY(); i++)
	{
		for(int j=0; j<wfcc->GetNumPOIsX(); j++)
		{
			file <<"[ "<< iPOIXY[i][j][0]<<", "<<iPOIXY[i][j][1]<<" ], "<<fU[i][j]<<", "<<fV[i][j]<<", "<<fZNCC[i][j]<<std::endl;
		}
	}

	file.close();

	std::cout << "CPU FFT-CC Time is: " << w.getElapsedTime() << std::endl;

	std::cout << iPOIXY[0][0][1] << ", " << iPOIXY[0][0][0] << ": " << fU[0][0] << ", " << fV[0][0] << ", " << fZNCC[0][0] << std::endl;

	wfcc->FinalizeFFTCC(iPOIXY,fU,fV,fZNCC);
}
//
////
////////TEST(Fftcc2D, Constructor)
////////{
////////	Fftcc2D * fftcc = new Fftcc2D(
////////		10,
////////		10);
////////
////////	Fftcc2D fftcc1 (Fftcc2D(10, 10));
////////
////////	EXPECT_EQ(fftcc->getNumPOIsX(), fftcc1.getNumPOIsY());
////////
////////	delete fftcc;
////////	fftcc = nullptr;
////////
////////}
//
////
////TEST(cuFFTCC2D_CPU, cuFFTCC2D_Copy_To_CPU)
////{
////	//!----------ROI based
////	cv::Mat mat = cv::imread("Example2\\crop_oht_cfrp_00.bmp ");
////
////	auto m_iWidth = mat.cols;
////	auto m_iHeight = mat.rows;
////
////	cv::Mat matnew(cv::Size(m_iWidth-2,m_iHeight-2),CV_8UC1);
////	cv::cvtColor(mat(cv::Range(1, m_iHeight - 1), cv::Range(1, m_iWidth - 1)), matnew, CV_BGR2GRAY);
////
////	std::cout << matnew.step << ", ";
////	std::cout << (float)matnew.data[10 * matnew.step + 20] << ", Kanzhege" << std::endl;
////
////	paDIC::cuFFTCC2D *fcc = new paDIC::cuFFTCC2D(matnew.cols, matnew.rows,
////		16, 16,
////		3, 3,
////		5, 5);
////	
////	real_t **iU, **iV;
////	real_t **fZNCC;
////
////	std::cout << fcc->GetNumPOIsX()<<", "<<fcc->GetNumPOIsY() << std::endl;
////
////	fcc->InitializeFFTCC(iU, iV, fZNCC, matnew);
////
////	cv::Mat mat1 = cv::imread("Example2\\crop_oht_cfrp_04.bmp");
////	//cv::Mat mat1new = mat1(cv::Range(1, m_iHeight - 1), cv::Range(1, m_iWidth - 1));
////	
////	cv::Mat matnew1(cv::Size(m_iWidth-2, m_iHeight-2), CV_8UC1);;
////
////	cv::cvtColor(mat1(cv::Range(1, m_iHeight - 1), cv::Range(1, m_iWidth - 1)), matnew1, CV_BGR2GRAY);
////
////
////	fcc->ComputeFFTCC(iU, iV, fZNCC, matnew1);
////
////	std::cout << iU[0][0] << ", " << iV[0][0] << ", " << fZNCC[0][0] << std::endl;
////
////
////	fcc->DestroyFFTCC(iU, iV, fZNCC);
////
////	delete fcc;
////	fcc = nullptr;
////
////	//!---------Whole Image based
////	cv::Mat wmat = cv::imread("Example2\\crop_oht_cfrp_00.bmp ");
////
////	auto wm_iWidth = mat.cols;
////	auto wm_iHeight = mat.rows;
////
////	cv::Mat wmatnew(cv::Size(wm_iWidth,wm_iHeight),CV_8UC1);
////	cv::cvtColor(wmat, wmatnew, CV_BGR2GRAY);
////
////	std::cout << wmatnew.step << ", ";
////	std::cout << (float)wmatnew.data[10 * wmatnew.step + 20] << ", Kanzhege" << std::endl;
////
////	paDIC::cuFFTCC2D *wfcc = new paDIC::cuFFTCC2D(wmatnew.cols, wmatnew.rows,
////		wmatnew.cols-2, wmatnew.rows-2,
////		1,1,
////		16, 16,
////		3, 3,
////		5, 5);
////	
////	real_t **wiU, **wiV;
////	real_t **wfZNCC;
////
////	std::cout << wfcc->GetNumPOIsX()<<", "<<wfcc->GetNumPOIsY() << std::endl;
////
////	wfcc->InitializeFFTCC(wiU, wiV, wfZNCC, wmatnew);
////
////	cv::Mat wmat1 = cv::imread("Example2\\crop_oht_cfrp_04.bmp");
////	//cv::Mat mat1new = mat1(cv::Range(1, m_iHeight - 1), cv::Range(1, m_iWidth - 1));
////	
////	cv::Mat wmatnew1(cv::Size(wm_iWidth, wm_iHeight), CV_8UC1);;
////
////	cv::cvtColor(wmat1, wmatnew1, CV_BGR2GRAY);
////
////
////	wfcc->ComputeFFTCC(wiU, wiV, wfZNCC, wmatnew1);
////
////	std::cout << wiU[0][0] << ", " << wiV[0][0] << ", " << wfZNCC[0][0] << std::endl;
////
////
////	wfcc->DestroyFFTCC(wiU, wiV, wfZNCC);
////
////	delete wfcc;
////	wfcc = nullptr;
////
////}
//
//
//TEST(cuFFTCC2D_GPU, cuFFTCC2D_StayOn_GPU)
//{
//	//!--------------ROI based
//	/*cv::Mat mat = cv::imread("Example2\\crop_oht_cfrp_00.bmp ");
//
//	auto m_iWidth = mat.cols;
//	auto m_iHeight = mat.rows;
//
//	cv::Mat matnew(cv::Size(m_iWidth-2,m_iHeight-2),CV_8UC1);
//
//	cv::cvtColor(mat(cv::Range(1, m_iHeight - 1), cv::Range(1, m_iWidth - 1)), matnew, CV_BGR2GRAY);
//
//	paDIC::cuFFTCC2D *fcc = new paDIC::cuFFTCC2D(matnew.cols, matnew.rows,
//		16, 16,
//		3, 3,
//		5, 5);
//
//	cv::Mat mat1 = cv::imread("Example2\\crop_oht_cfrp_04.bmp ");
//	cv::Mat matnew1(cv::Size(m_iWidth-2,m_iHeight-2),CV_8UC1);
//	cv::cvtColor(mat1(cv::Range(1, m_iHeight - 1), cv::Range(1, m_iWidth - 1)), matnew1, CV_BGR2GRAY);
//
//
//	real_t **iU, **iV;
//	real_t **fZNCC;
//
//	real_t *idU, *idV;
//	real_t *fdZNCC;
//
//	fcc->cuInitializeFFTCC(idU,idV,fdZNCC,matnew);
//
//	fcc->ResetRefImg(matnew);
//
//	fcc->cuComputeFFTCC(idU,idV,fdZNCC,matnew1);
//
//	hcreateptr<real_t>(iU, fcc->GetNumPOIsY(), fcc->GetNumPOIsX());
//	hcreateptr<real_t>(iV, fcc->GetNumPOIsY(), fcc->GetNumPOIsX());
//	hcreateptr<real_t>(fZNCC, fcc->GetNumPOIsY(), fcc->GetNumPOIsX());
//
//	cudaMemcpy(iU[0], idU, sizeof(int_t)*fcc->GetNumPOIs(),cudaMemcpyDeviceToHost);
//	cudaMemcpy(iV[0], idV, sizeof(int_t)*fcc->GetNumPOIs(),cudaMemcpyDeviceToHost);
//	cudaMemcpy(fZNCC[0], fdZNCC, sizeof(int_t)*fcc->GetNumPOIs(),cudaMemcpyDeviceToHost);
//
//	std::cout << iU[0][0] << ", " << iV[0][0] << ", " << fZNCC[0][0] << std::endl;
//
//	fcc->cuDestroyFFTCC(idU,idV,fdZNCC);
//
//
//	delete fcc;
//	fcc = nullptr;
//
//	
//	hdestroyptr(iU);
//	hdestroyptr(iV);
//	hdestroyptr(fZNCC);*/
//
//	//!--------------Image based
//	cv::Mat wmat = cv::imread("Example1\\fu_0.bmp");
//
//	auto wm_iWidth = wmat.cols;
//	auto wm_iHeight = wmat.rows;
//
//	cv::Mat wmatnew(cv::Size(wm_iWidth,wm_iHeight),CV_8UC1);
//
//	cv::cvtColor(wmat, wmatnew, CV_BGR2GRAY);
//
//	paDIC::cuFFTCC2D *wfcc = new paDIC::cuFFTCC2D(wmatnew.cols, wmatnew.rows,
//		wmatnew.cols-2, wmatnew.rows-2,
//		1,1,
//		16, 16,
//		5, 5,
//		10, 10);
//
//	cv::Mat wmat1 = cv::imread("Example1\\fu_1.bmp");
//	cv::Mat wmatnew1(cv::Size(wm_iWidth,wm_iHeight),CV_8UC1);
//	cv::cvtColor(wmat1, wmatnew1, CV_BGR2GRAY);
//
//
//	real_t **wiU, **wiV;
//	real_t **wfZNCC;
//
//	real_t *widU, *widV;
//	real_t *wfdZNCC;
//
//	wfcc->cuInitializeFFTCC(widU,widV,wfdZNCC,wmatnew);
//
//	wfcc->ResetRefImg(wmatnew);
//
//
//	wfcc->cuComputeFFTCC(widU,widV,wfdZNCC,wmatnew1);
//
//	hcreateptr<real_t>(wiU, wfcc->GetNumPOIsY(), wfcc->GetNumPOIsX());
//	hcreateptr<real_t>(wiV, wfcc->GetNumPOIsY(), wfcc->GetNumPOIsX());
//	hcreateptr<real_t>(wfZNCC, wfcc->GetNumPOIsY(), wfcc->GetNumPOIsX());
//
//	cudaMemcpy(wiU[0], widU, sizeof(int_t)*wfcc->GetNumPOIs(),cudaMemcpyDeviceToHost);
//	cudaMemcpy(wiV[0], widV, sizeof(int_t)*wfcc->GetNumPOIs(),cudaMemcpyDeviceToHost);
//	cudaMemcpy(wfZNCC[0], wfdZNCC, sizeof(int_t)*wfcc->GetNumPOIs(),cudaMemcpyDeviceToHost);
//
//	std::cout << wiU[0][0] << ", " << wiV[0][0] << ", " << wfZNCC[0][0] << std::endl;
//
//	wfcc->cuDestroyFFTCC(widU,widV,wfdZNCC);
//
//
//	delete wfcc;
//	wfcc = nullptr;
//	
//	hdestroyptr(wiU);
//	hdestroyptr(wiV);
//	hdestroyptr(wfZNCC);
//}