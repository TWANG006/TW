#include "icgnworkerthread.h"
#include <omp.h>

ICGNWorkerThread::ICGNWorkerThread(
	ImageBufferPtr refImgBuffer,
	ImageBufferPtr tarImgBuffer,
	VecBufferfPtr fUBuffer,
	VecBufferfPtr fVBuffer,
	VecBufferiPtr iPOIXYBuffer,
	const QRect &roi,
	const int iNumICGNThreads,
	int iWidth, int iHeight,
	int iNumberX, int iNumberY,
	int iSubsetX, int iSubsetY,
	int iNumbIterations,
	float fDeltaP)
	: m_ICGN2DPtr(nullptr)
	, m_refImgBuffer(refImgBuffer)
	, m_tarImgBuffer(tarImgBuffer)
	, m_fUBuffer(fUBuffer)
	, m_fVBuffer(fVBuffer)
	, m_iPOIXYBuffer(iPOIXYBuffer)
	, m_iNumberIterations(iNumberX*iNumberY)
{
	omp_set_num_threads(iNumICGNThreads);

	m_ICGN2DPtr.reset(new ICGN2D_CPU(
		iWidth, iHeight,
		roi.x(), roi.y(),
		roi.width(), roi.height(),
		iSubsetX, iSubsetY,
		iNumberX, iNumberY,
		20,
		0.001,
		TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
		TW::paDIC::paDICThreadFlag::Single));
}

ICGNWorkerThread::~ICGNWorkerThread()
{
	std::cout<<"ICGN Thread is stopped and deleted"<<std::endl;
}

void ICGNWorkerThread::processFrame()
{
	m_refImgBuffer->DeQueue(refImg);
	m_ICGN2DPtr->ResetRefImg(refImg);

	m_tarImgBuffer->DeQueue(tarImg);
	m_fUBuffer->DeQueue(fU);
	m_fVBuffer->DeQueue(fV);
	m_iPOIXYBuffer->DeQueue(iPOIXY);

	// For debug the image intensity variations
	/*cv::imwrite("1.bmp", refImg);
	cv::imwrite("2.bmp", tarImg);

	for (int i = -10; i < 10; i++)
	{
		std::cout << int(refImg.at<uchar>(iPOIXY[1] + i, iPOIXY[0] + i)) << ", " <<
			int(tarImg.at<uchar>(iPOIXY[1] + i, iPOIXY[0] + i)) << std::endl;
	}*/


	/*std::cout<<refImg.rows<<", "<<refImg.cols<<", "<<(refImg.at<uchar>(iPOIXY[1], iPOIXY[0]))<<std::endl;
	std::cout<<tarImg.rows<<", "<<tarImg.cols<<", "<<(tarImg.at<uchar>(iPOIXY[1], iPOIXY[0]))<<std::endl;*/

	m_ICGN2DPtr->ICGN2D_Algorithm(
		fU.data(),
		fV.data(),
		m_iNumberIterations.data(),
		iPOIXY.data(),
		tarImg);

	/*std::cout<<iPOIXY[1]<<", "<<iPOIXY[0]<<std::endl;*/
	//	float j = 0;
	//#pragma omp parallel for
	//	for (int i = 0; i < 1000000; i++)
	//	{
	//		j += 0.5;
	//	}
	/*emit testSignal(fU[0]);*/
}