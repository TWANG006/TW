#include "fftcctworkerthread.h"
#include <QDebug>

FFTCCTWorkerThread::FFTCCTWorkerThread(ImageBufferPtr refImgBuffer,
									   ImageBufferPtr tarImgBuffer,
									   int iWidth, int iHeight,
									   int iSubsetX, int iSubsetY,
									   int iGridSpaceX, int iGridSpaceY,
									   int iMarginX, int iMarginY,
									   const QRect &roi,
									   const cv::Mat &firstFrame,
									   int *&d_iU,
									   int *&d_iV,
									   float *&d_fZNCC)
	: m_refImgBuffer(refImgBuffer)
	, m_tarImgBuffer(tarImgBuffer)
	, m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_ROI(roi)
	, m_Fftcc2DPtr(nullptr)
	, m_d_iU(d_iU)
	, m_d_iV(d_iV)
	, m_d_fZNCC(d_fZNCC)
{
	// Do the initialization for the paDIC's cuFFTCC here in the constructor
	// 1. Construct the cuFFTCC2D object using the whole image
	m_Fftcc2DPtr.reset(new TW::paDIC::cuFFTCC2D(iWidth, iHeight,
												m_ROI.width(), m_ROI.height(),
												m_ROI.x(), m_ROI.y(),
												iSubsetX, iSubsetY,
												iGridSpaceX, iGridSpaceY,
												iMarginX, iMarginY));
	
	//2. Do the initialization for cuFFTCC2D object
	m_Fftcc2DPtr->cuInitializeFFTCC(m_d_iU, m_d_iV, m_d_fZNCC, firstFrame);
}

FFTCCTWorkerThread::~FFTCCTWorkerThread()
{
	m_Fftcc2DPtr->cuDestroyFFTCC(m_d_iU, m_d_iV, m_d_fZNCC);
}

void FFTCCTWorkerThread::processFrame(const int &iFrameCount)
{
	cv::Mat tempImg;
	cv::Mat tarImg;

	// 1. Every 50 frames updates the reference image
	if(iFrameCount % 50 ==1)
	{
		m_refImgBuffer->DeQueue(tempImg);
		m_Fftcc2DPtr->ResetRefImg(tempImg);
		qDebug()<<"ref";
	}
		
	m_tarImgBuffer->DeQueue(tarImg);

	// 2. Do the FFTCC computation
	m_Fftcc2DPtr->cuComputeFFTCC(m_d_iU, m_d_iV, m_d_fZNCC, tarImg);
	qDebug()<<"tar";

	// 3. Use the results to update the POI positions

	// 4. Calculate the color map for the iU and iV images


	// 5. Invoke the CUDA & OpenGL interoperability
	// 5.1 Map the target image data
	// 5.2 Map the colormap data
	// 5.3 Normalize to [0,1] scale


}