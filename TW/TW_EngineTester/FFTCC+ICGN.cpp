#include "TW_utils.h"
#include "TW_MemManager.h"
#include "TW_StopWatch.h"
#include "TW_paDIC_FFTCC2D_CPU.h"
#include "TW_paDIC_ICGN2D_CPU.h"

#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <omp.h>
#include <gtest\gtest.h>
#include <fstream>

using namespace std;
using namespace TW;

TEST(FFTCC_ICGN, CPU_Multicore)
{
	omp_set_num_threads(6);

	float aveFFTCC = 0;
	float aveICGN = 0;

	ofstream textfile;

	int marginX = 3;
	int marginY = 3;

	int ROIx = 30;
	int ROIy = 30;

	int startX = 242;
	int startY = 242;

	textfile.open("result.csv", ios::out | ios::trunc);

	{
		std::cout << "====================1====================: " << std::endl;
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_1.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;
	}
	{
		std::cout << "====================2====================: " << std::endl;
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_2.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;	}

	std::cout << "====================3====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_3.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;	}

	std::cout << "====================4====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_4.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;	}

	std::cout << "====================5====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_5.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}
		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;	}

	std::cout << "====================6====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_6.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			aveICGN += 1000 * (end - start);

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;
	}

	std::cout << "====================7====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_7.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;
	}

	std::cout << "====================8====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_8.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;	}
	std::cout << "====================9====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_9.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;	}
	std::cout << "====================10====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_10.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;	}
	std::cout << "====================11====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_11.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;	}
	std::cout << "====================12====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_12.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}
		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;	}

	std::cout << "====================13====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_13.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;	}

	std::cout << "====================14====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_14.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;	}

	std::cout << "====================15====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_15.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;	}

	std::cout << "====================16====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_16.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;}

	std::cout << "====================17====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_17.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;	}

	std::cout << "====================18====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_18.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}
		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;	}

	std::cout << "====================19====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

			cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
			cv::Mat Tmat = cv::imread("Example1\\fu_19.bmp");

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;}
	std::cout << "====================20====================: " << std::endl;
	{
		aveFFTCC = 0;
		aveICGN = 0;
		for (int i = 0; i < 5; i++)
		{

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
				startX, startY,
				ROIx, ROIy,
				10, 10,
				3, 3,
				marginX, marginY,
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
			//std::cout << "CPU FFT-CC Time is: " << 1000 * (end - start) << std::endl;

			aveFFTCC += 1000 * (end - start);

			int *iters;
			hcreateptr(iters, wfcc->GetNumPOIs());

			TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
				startX, startY,
				ROIx, ROIy,
				10, 10,
				wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
				20,
				0.001f,
				TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
				TW::paDIC::paDICThreadFlag::Multicore);

			icgn->ResetRefImg(Rmatnew);

			float fPreTime = 0, fICGNTime = 0;
			icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew, fPreTime, fICGNTime);

			aveICGN += fICGNTime;

			hdestroyptr(fU);
			hdestroyptr(fV);
			hdestroyptr(fZNCC);
			hdestroyptr(iters);

			delete icgn;
			delete wfcc;
		}

		textfile << aveFFTCC / 5.0f << "," << aveICGN / 5.0f << endl;}
	textfile.close();
}
//
//TEST(FFTCC_ICGN, CPU_Single)
//{
//	omp_set_num_threads(12);
//
//	cv::Mat Rmat = cv::imread("Example1\\fu_0.bmp");
//	cv::Mat Tmat = cv::imread("Example1\\fu_20.bmp");
//
//	auto wm_iWidth = Rmat.cols;
//	auto wm_iHeight = Rmat.rows;
//
//	cv::Mat Rmatnew(cv::Size(wm_iWidth, wm_iHeight), CV_8UC1);
//	cv::Mat Tmatnew(cv::Size(wm_iWidth, wm_iHeight), CV_8UC1);
//
//	cv::cvtColor(Rmat, Rmatnew, CV_BGR2GRAY);
//	cv::cvtColor(Tmat, Tmatnew, CV_BGR2GRAY);
//
//	int*** iPOIXY;
//	float **fU, **fV, **fZNCC;
//
//	paDIC::Fftcc2D_CPU *wfcc = new paDIC::Fftcc2D_CPU(
//		Rmatnew.cols, Rmatnew.rows,
//		98, 98,
//		324, 324,
//		10, 10,
//		3, 3,
//		3, 3,
//		TW::paDIC::paDICThreadFlag::Single);
//
//	wfcc->InitializeFFTCC(
//		Rmatnew,
//		iPOIXY,
//		fU,
//		fV,
//		fZNCC);
//
//	std::cout << "Number of POIs: " << wfcc->GetNumPOIs() << std::endl;
//
//	StopWatch w;
//	w.start();
//	wfcc->Algorithm_FFTCC(
//		Tmatnew,
//		iPOIXY,
//		fU,
//		fV,
//		fZNCC);
//	w.stop();
//
//	//std::cout << "CPU FFT-CC Time is: " << w.getElapsedTime() << std::endl;
//
//	int *iters;
//	hcreateptr(iters, wfcc->GetNumPOIs());
//
//	TW::paDIC::ICGN2D_CPU *icgn = new paDIC::ICGN2D_CPU(wm_iWidth, wm_iHeight,
//		98, 98,
//		324, 324,
//		10, 10,
//		wfcc->GetNumPOIsX(), wfcc->GetNumPOIsY(),
//		20,
//		0.001f,
//		TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
//		TW::paDIC::paDICThreadFlag::Single);
//
//	icgn->ResetRefImg(Rmatnew);
//
//	StopWatch w1;
//	w1.start();
//	icgn->ICGN2D_Algorithm(fU[0], fV[0], iters, iPOIXY[0][0], Tmatnew);
//	w1.stop();
//
//	std::cout << "CPU ICGNTime is: " << w1.getElapsedTime() << std::endl;
//	//std::cout << fU[0][0] << ", " << fV[0][0] << std::endl;
//
//	hdestroyptr(fU);
//	hdestroyptr(fV);
//	hdestroyptr(fZNCC);
//	hdestroyptr(iters);
//
//	delete icgn;
//	delete wfcc;
//}
