#include "TW_utils.h"
#include "TW_MemManager.h"
#include "TW_StopWatch.h"
#include "TW_paDIC_FFTCC2D_CPU.h"
#include "TW_paDIC_ICGN2D_CPU.h"

#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <omp.h>
#include <gtest\gtest.h>

using namespace TW;

TEST(FFTCC_ICGN, CPU_Multicore)
{
	omp_set_num_threads(12);

	cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
	cv::Mat Tmat = cv::imread("Example1\\fu_20.bmp");

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
		98, 98,
		324, 324,
		10, 10,
		3, 3,
		3, 3,
		TW::paDIC::paDICThreadFlag::Multicore);

	wfcc->InitializeFFTCC(
		Rmatnew,
		iPOIXY,
		fU,
		fV,
		fZNCC);

	std::cout << "Number of POIs: " << wfcc->GetNumPOIs() << std::endl;

	/*StopWatch w;
	w.start();*/
	
	double start = omp_get_wtime();
	wfcc->Algorithm_FFTCC(
		Tmatnew,
		iPOIXY,
		fU,
		fV,
		fZNCC);
	/*w.stop();*/

	double end = omp_get_wtime();
	//std::cout << "CPU FFT-CC Time is: " << w.getElapsedTime() << std::endl;
	std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

	int *iters;
	hcreateptr(iters, wfcc->GetNumPOIs());

	TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
		98, 98,
		324, 324,
		10, 10,
		wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
		20,
		0.001f,
		TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
		TW::paDIC::paDICThreadFlag::Multicore);

	icgn->ResetRefImg(Rmatnew);

	/*StopWatch w1;
	w1.start();*/
	start = omp_get_wtime();
	icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew);
	/*w1.stop();*/
	end = omp_get_wtime();

	std::cout << "CPU ICGN Time is: " <<1000 * (end - start)  /*w1.getElapsedTime()*/ << std::endl;
	std::cout << fU[0][0] << ", " << fV[0][0] << std::endl;

	hdestroyptr(fU);
	hdestroyptr(fV);
	hdestroyptr(fZNCC);
	hdestroyptr(iters);

	delete icgn;
	delete wfcc;
}

TEST(FFTCC_ICGN, CPU_Single)
{
	omp_set_num_threads(12);

	cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
	cv::Mat Tmat = cv::imread("Example1\\fu_20.bmp");

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
		98, 98,
		324, 324,
		10, 10,
		3, 3,
		3, 3,
		TW::paDIC::paDICThreadFlag::Single);

	wfcc->InitializeFFTCC(
		Rmatnew,
		iPOIXY,
		fU,
		fV,
		fZNCC);

	std::cout << "Number of POIs: " << wfcc->GetNumPOIs() << std::endl;

	StopWatch w;
	w.start();
	wfcc->Algorithm_FFTCC(
		Tmatnew,
		iPOIXY,
		fU,
		fV,
		fZNCC);
	w.stop();

	std::cout << "CPU FFT-CC Time is: " << w.getElapsedTime() << std::endl;

	int *iters;
	hcreateptr(iters, wfcc->GetNumPOIs());

	TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
		98, 98,
		324, 324,
		10, 10,
		wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
		20,
		0.001f,
		TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
		TW::paDIC::paDICThreadFlag::Single);

	icgn->ResetRefImg(Rmatnew);

	StopWatch w1;
	w1.start();
	icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew);
	w1.stop();

	std::cout << "CPU ICGN Time is: " << w1.getElapsedTime() << std::endl;
	std::cout << fU[0][0] << ", " << fV[0][0] << std::endl;

	hdestroyptr(fU);
	hdestroyptr(fV);
	hdestroyptr(fZNCC);
	hdestroyptr(iters);

	delete icgn;
	delete wfcc;
}
