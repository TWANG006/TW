//#include "TW_paDIC_FFTCC2D.h"
//
//#include <gtest\gtest.h>
//
//using namespace TW::paDIC;
//
//TEST(Fftcc2D, Constructor)
//{
//	Fftcc2D * fftcc = new Fftcc2D(
//		10,
//		10);
//
//	Fftcc2D fftcc1 (Fftcc2D(10, 10));
//
//	EXPECT_EQ(fftcc->getNumPOIsX(), fftcc1.getNumPOIsY());
//
//	delete fftcc;
//	fftcc = nullptr;
//
//}
#include "TW_paDIC_cuFFTCC2D.h"
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <gtest\gtest.h>

using namespace TW;

TEST(cuFFTCC2D1, cuFFTCC2D)
{
	cv::Mat mat = cv::imread("Example1\\fu_0.bmp", CV_8UC1);

	auto m_iWidth = mat.cols;
	auto m_iHeight = mat.rows;

	cv::Mat matnew = mat(cv::Range(1, m_iHeight - 2), cv::Range(1, m_iWidth - 2));


	paDIC::cuFFTCC2D *fcc = new paDIC::cuFFTCC2D(m_iWidth - 2, m_iHeight - 2,
		16,16,
		3,3,
		5,5);
	
	int_t **iU, **iV;
	real_t **fZNCC;

	std::cout << fcc->GetNumPOIsX()<<", "<<fcc->GetNumPOIsY() << std::endl;

	fcc->InitializeFFTCC(iU, iV, fZNCC, matnew);

	cv::Mat mat1 = cv::imread("Example1\\fu_20.bmp", CV_8UC1);
	cv::Mat mat1new = mat1(cv::Range(1, m_iHeight - 2), cv::Range(1, m_iWidth - 2));

	fcc->ComputeFFTCC(iU, iV, fZNCC, mat1new);

	for (int i = 0; i < fcc->GetNumPOIsY(); i++)
	{
		for (int j = 0; j < fcc->GetNumPOIsX(); j++)
		{
			std::cout << iU[i][j] << ", " << iV[i][j] << ", " << fZNCC[i][j] << std::endl;
		}
	}


	fcc->DestroyFFTCC();


	delete fcc;
	fcc = nullptr;
}