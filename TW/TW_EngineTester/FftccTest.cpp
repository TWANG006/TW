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
#include "TW_utils.h"
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <gtest\gtest.h>

using namespace TW;

TEST(cuFFTCC2D1, cuFFTCC2D)
{
	cv::Mat mat = cv::imread("Example2\\crop_oht_cfrp_00.bmp ");

	auto m_iWidth = mat.cols;
	auto m_iHeight = mat.rows;

	cv::Mat matnew(cv::Size(m_iWidth-2,m_iHeight-2),CV_8UC1);

	cv::cvtColor(mat(cv::Range(1, m_iHeight - 1), cv::Range(1, m_iWidth - 1)), matnew, CV_BGR2GRAY);

	// = mat(cv::Range(1, m_iHeight - 1), cv::Range(1, m_iWidth - 1));

	//cv::imwrite("New_ref.bmp", matnew);

	std::cout << matnew.step << ", ";
	std::cout << (float)matnew.data[10 * matnew.step + 20] << ", Kanzhege" << std::endl;

	paDIC::cuFFTCC2D *fcc = new paDIC::cuFFTCC2D(matnew.cols, matnew.rows,
		16, 16,
		3, 3,
		5, 5);
	
	int_t **iU, **iV;
	real_t **fZNCC;

	std::cout << fcc->GetNumPOIsX()<<", "<<fcc->GetNumPOIsY() << std::endl;

	fcc->InitializeFFTCC(iU, iV, fZNCC, matnew);

	cv::Mat mat1 = cv::imread("Example2\\crop_oht_cfrp_04.bmp");
	//cv::Mat mat1new = mat1(cv::Range(1, m_iHeight - 1), cv::Range(1, m_iWidth - 1));
	
	cv::Mat matnew1(cv::Size(m_iWidth-2, m_iHeight-2), CV_8UC1);;

	cv::cvtColor(mat1(cv::Range(1, m_iHeight - 1), cv::Range(1, m_iWidth - 1)), matnew1, CV_BGR2GRAY);


	fcc->ComputeFFTCC(iU, iV, fZNCC, matnew1);

	std::cout << fZNCC[0] << std::endl;

	for (int i = 0; i < fcc->GetNumPOIsY(); i++)
	{
		for (int j = 0; j < fcc->GetNumPOIsX(); j++)
		{
			std::cout << iU[0][0] << ", " << iV[0][0] << ", " << fZNCC[0][0] << std::endl;
		}
	}


	fcc->DestroyFFTCC(iU, iV, fZNCC);

	delete fcc;
	fcc = nullptr;
}