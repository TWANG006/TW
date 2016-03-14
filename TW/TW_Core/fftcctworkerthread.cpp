#include "fftcctworkerthread.h"

FFTCCTWorkerThread::FFTCCTWorkerThread(ImageBufferPtr refImgBuffer,
									   ImageBufferPtr tarImgBuffer,
									   int iWidth, int iHeight,
									   int iSubsetX, int iSubsetY,
									   int iGridSpaceX, int iGridSpaceY,
									   int iMarginX, int iMarginY,
									   const QRect &roi,
									   const cv::Mat &firstFrame,
									   QObject *parent)
	: m_refImgBuffer(refImgBuffer)
	, m_tarImgBuffer(tarImgBuffer)
	, m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_ROI(roi)
	, m_Fftcc2DPtr(nullptr)
	, QObject(parent)
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
	
}

FFTCCTWorkerThread::~FFTCCTWorkerThread()
{
	
}

void FFTCCTWorkerThread::processFrame(const int &iFrameCount)
{
	// Every 50 frames updates the reference image
	if(iFrameCount % 50 ==1)
		;

	// Do the FFTCC 

	// Use the results to update the POI positions

	// Invoke the CUDA & OpenGL interoperability

}