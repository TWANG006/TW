#include "TW_paDIC_cuFFTCC2D.h"
#include "TW_utils.h"
#include "TW_MemManager.h"
#include "TW_paDIC_ICGN2D_CPU.h"
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <gtest\gtest.h>

using namespace TW;

TEST(Gradient, Gradient_s)
{
	cv::Mat mat = cv::imread("Example2\\crop_oht_cfrp_00.bmp");

	auto imgWidth = mat.cols;
	auto imgHeight= mat.rows;

	cv::Mat matS(cv::Size(imgWidth, imgHeight), CV_8UC1);
	cv::cvtColor(mat, matS, CV_BGR2GRAY);

	float **Gx, **Gy/*, *Gxy*/;

	hcreateptr(Gx, (imgHeight-2), (imgWidth-2));
	hcreateptr(Gy, (imgHeight-2), (imgWidth-2));
	//hcreateptr(Gxy, (imgWidth-2)*(imgHeight-2));

	try{
		Gradient_s(matS, 1,1,imgWidth-2, imgHeight-2, imgWidth, imgHeight, TW::Quadratic, Gx, Gy/*, Gxy*/);
	}
	catch(const char* c)
	{
		std::cerr<<c<<std::endl;
	}
	std::cout<<(float)matS.at<uchar>(0,1)<<", "<<(float)matS.at<uchar>(0,2)<<", "<<(float)matS.at<uchar>(0,3)<<std::endl;
	std::cout<<(float)matS.at<uchar>(1,1)<<", "<<(float)matS.at<uchar>(1,2)<<", "<<(float)matS.at<uchar>(1,3)<<std::endl;
	std::cout<<(float)matS.at<uchar>(2,1)<<", "<<(float)matS.at<uchar>(2,2)<<", "<<(float)matS.at<uchar>(2,3)<<std::endl;
	std::cout<<"GradientX: "<<Gx[0][1]<<", "<<"GradientY: "<<Gy[0][1]<<std::endl;

	hdestroyptr(Gx);
	hdestroyptr(Gy);
	//hdestroyptr(Gxy);
}

TEST(BSpline, BSplineInterpolation)
{                                                  
	cv::Mat mat = cv::imread("Example2\\crop_oht_cfrp_01.bmp");

	auto imgWidth = mat.cols;
	auto imgHeight= mat.rows;

	cv::Mat matS(cv::Size(imgWidth, imgHeight), CV_8UC1);
	cv::cvtColor(mat, matS, CV_BGR2GRAY);

	float****fBSpline;

	hcreateptr(fBSpline, imgHeight-4, imgWidth-4, 4, 4);
	BicubicSplineCoefficients_s(matS, 2, 2, imgWidth-4, imgHeight-4, imgWidth, imgHeight, fBSpline);


	std::cout<<"First: "<<std::endl;
	for(int i=0;i<4;i++)
	{
		for(int j=0;j<4;j++)
		{
			std::cout<<fBSpline[0][0][i][j]<<", ";
		}
		std::cout<<std::endl;
	}
	hdestroyptr(fBSpline);
}

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


	TW::paDIC::ICGN2D_CPU icgn(matR,matT,2,2,imgWidth-4,imgHeight-4,16,16,92,92,20,0.001);
	icgn.ICGN2D_Precomputation_Prepare();
	icgn.ICGN2D_Precomputation();

	icgn.ICGN2D_Prepare();
	float u=0, v=0;
	int iter = 0;
	icgn.ICGN2D_Compute(u,v, iter, 28,28,0);

	std::cout<<"The displacement is [" << u << ", " << v << "]\n";

	icgn.ICGN2D_Precomputation_Finalize();
	icgn.ICGN2D_Finalize();
}