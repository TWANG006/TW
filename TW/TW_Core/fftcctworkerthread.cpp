#include "fftcctworkerthread.h"
#include <QDebug>

#include "cuda_utils.cuh"

FFTCCTWorkerThread::FFTCCTWorkerThread(ImageBufferPtr refImgBuffer,
									   ImageBufferPtr tarImgBuffer,
									   int iWidth, int iHeight,
									   int iSubsetX, int iSubsetY,
									   int iGridSpaceX, int iGridSpaceY,
									   int iMarginX, int iMarginY,
									   const QRect &roi,
									   const cv::Mat &firstFrame,
									   std::shared_ptr<SharedResources>& s)
	: m_refImgBuffer(refImgBuffer)
	, m_tarImgBuffer(tarImgBuffer)
	, m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_ROI(roi)
	, m_Fftcc2DPtr(nullptr)
	, m_sharedResources(s)
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
	m_Fftcc2DPtr->cuInitializeFFTCC(m_d_fU, m_d_fV, m_d_fZNCC, firstFrame);
}

FFTCCTWorkerThread::~FFTCCTWorkerThread()
{
	m_Fftcc2DPtr->cuDestroyFFTCC(m_d_fU, m_d_fV, m_d_fZNCC);

	cudaDeviceReset();

	deleteObject(m_sharedResources->sharedContext);
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

		// 3.1 Use the results to update the POI positions

		// 3.2 TODO: Copy the iU, iV to host memory for ICGN

		// qDebug()<<"ref";
	}
		
	m_tarImgBuffer->DeQueue(tarImg);

	// 2. Do the FFTCC computation
	m_Fftcc2DPtr->cuComputeFFTCC(m_d_fU, m_d_fV, m_d_fZNCC, tarImg);

	/*float *i = new float;
	cudaMemcpy(i, &m_d_fZNCC[0], sizeof(float), cudaMemcpyDeviceToHost);

	qDebug()<<"tar"<<"  "<<*i;*/
	/*static const unsigned char texture_data[] =
	{
		0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00,
		0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
		0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00,
		0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
		0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00,
		0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
		0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00,
		0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF
	};*/

	if(m_sharedResources->sharedTexture!=nullptr &&
	   m_sharedResources->sharedContext!=nullptr &&
	   m_sharedResources->sharedProgram!=nullptr)
	{
		m_sharedResources->sharedContext->makeCurrent(m_sharedResources->sharedSurface);

		m_sharedResources->sharedProgram->bind();
		m_sharedResources->sharedTexture->bind();

		//m_sharedResources->sharedTexture->setData(0, QOpenGLTexture::Red,QOpenGLTexture::UInt8,texture_data);

		checkCudaErrors(cudaMemcpyToArray(m_sharedResources->cudaImgArray,
			0,
			0,
			m_Fftcc2DPtr->m_cuHandle.m_d_fTarImg,
			640 * 480,
			cudaMemcpyDeviceToDevice));

		m_sharedResources->sharedTexture->release();
		m_sharedResources->sharedProgram->release();

		m_sharedResources->sharedContext->doneCurrent();

		emit frameReady();
	}
	// 4. Calculate the color map for the iU and iV images


	// 5. Invoke the CUDA & OpenGL interoperability
	// 5.1 Map the target image data
	// 5.2 Map the colormap data
	// 5.3 Normalize to [0,1] scale
	// delete i; i = nullptr;
}

void FFTCCTWorkerThread::render()
{

}