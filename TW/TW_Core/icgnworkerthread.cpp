#include "icgnworkerthread.h"
#include <omp.h>

ICGNWorkerThread::ICGNWorkerThread(
	ImageBufferPtr refImgBuffer,
	ImageBufferPtr tarImgBuffer,
	VecBufferfPtr fUBuffer,
	VecBufferfPtr fVBuffer,
	VecBufferiPtr iPOIXYBuffer,
	const QRect &roi,
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
	omp_set_num_threads(4);

	m_ICGN2DPtr.reset(new ICGN2D_CPU(
		iWidth, iHeight,
		roi.x(), roi.y(),
		roi.width(), roi.height(),
		iNumberX, iNumberY,
		iSubsetX, iSubsetY,
		20,
		0.001,
		TW::paDIC::ICGN2DInterpolationFLag::Bicubic,
		TW::paDIC::paDICThreadFlag::Multicore));
}

ICGNWorkerThread::~ICGNWorkerThread()
{
	
}

void ICGNWorkerThread::processFrame()
{
	cv::Mat tempImg;
	std::vector<float> fU;
	std::vector<float> fV;
	std::vector<int> iPOIXY;
	m_refImgBuffer->DeQueue(tempImg);
	m_ICGN2DPtr->ResetRefImg(tempImg);


	m_tarImgBuffer->DeQueue(tempImg);
	m_fUBuffer->DeQueue(fU);
	m_fVBuffer->DeQueue(fV);
	m_iPOIXYBuffer->DeQueue(iPOIXY);

	m_ICGN2DPtr->ICGN2D_Algorithm(
		fU.data(),
		fV.data(),
		m_iNumberIterations.data(),
		iPOIXY.data(),
		tempImg);

	std::cout<<fU[0]<<", "<<fV[0]<<std::endl;
//	float j = 0;
//#pragma omp parallel for
//	for (int i = 0; i < 1000000; i++)
//	{
//		j += 0.5;
//	}
	emit testSignal(fU[0]);
}