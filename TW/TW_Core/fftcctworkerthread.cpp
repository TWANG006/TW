#include "fftcctworkerthread.h"
#include <QDebug>

#include "cuda_utils.cuh"
#include "TW_MemManager.h"

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
	, m_d_fU(nullptr)
	, m_d_fV(nullptr)
	, m_d_fAccumulateU(nullptr)
	, m_d_fAccumulateV(nullptr)
	, m_d_fMaxU(nullptr)
	, m_d_fMinU(nullptr)
	, m_d_fMaxV(nullptr)
	, m_d_fMinV(nullptr)
	, m_d_iCurrentPOIXY(nullptr)
	, m_d_fZNCC(nullptr)
{
	// Do the initialization for the paDIC's cuFFTCC here in the constructor
	// 1. Construct the cuFFTCC2D object using the whole image
	m_Fftcc2DPtr.reset(new TW::paDIC::cuFFTCC2D(iWidth, iHeight,
												m_ROI.width(), m_ROI.height(),
												m_ROI.x(), m_ROI.y(),
												iSubsetX, iSubsetY,
												iGridSpaceX, iGridSpaceY,
												iMarginX, iMarginY));
	
	m_iNumberX = m_Fftcc2DPtr->GetNumPOIsX();
	m_iNumberY = m_Fftcc2DPtr->GetNumPOIsY();

	// Allocate memory for the current U & V
	m_iNumPOIs = m_Fftcc2DPtr->GetNumPOIs();
	cudaMalloc((void**)&m_d_fAccumulateU, sizeof(TW::real_t)*m_iNumPOIs);
	cudaMalloc((void**)&m_d_fAccumulateV, sizeof(TW::real_t)*m_iNumPOIs);

	// Initialize the current U & V to 0


	//2. Do the initialization for cuFFTCC2D object
	cudaMalloc((void**)&m_d_iCurrentPOIXY, sizeof(int)*m_iNumPOIs * 2);
	m_Fftcc2DPtr->cuInitializeFFTCC(m_d_fU, m_d_fV, m_d_fZNCC, firstFrame);
	cudaMemcpy(m_d_iCurrentPOIXY, m_Fftcc2DPtr->g_cuHandle.m_d_iPOIXY, sizeof(int)*m_iNumPOIs * 2, cudaMemcpyDeviceToDevice);

	//3. Allocate memory for the max and min [U,V]'s
	cudaMalloc((void**)&m_d_fMaxU, sizeof(TW::real_t));
	cudaMalloc((void**)&m_d_fMinU, sizeof(TW::real_t));
	cudaMalloc((void**)&m_d_fMaxV, sizeof(TW::real_t));
	cudaMalloc((void**)&m_d_fMinV, sizeof(TW::real_t));
}

FFTCCTWorkerThread::~FFTCCTWorkerThread()
{
	m_Fftcc2DPtr->cuDestroyFFTCC(m_d_fU, m_d_fV, m_d_fZNCC);

	TW::cudaSafeFree(m_d_fAccumulateU);
	TW::cudaSafeFree(m_d_fAccumulateV);
	TW::cudaSafeFree(m_d_fMaxU);
	TW::cudaSafeFree(m_d_fMinU);
	TW::cudaSafeFree(m_d_fMaxV);
	TW::cudaSafeFree(m_d_fMaxV);
	TW::cudaSafeFree(m_d_iCurrentPOIXY);

	cudaDeviceReset();

	deleteObject(m_sharedResources->sharedContext);
}




void FFTCCTWorkerThread::processFrame(const int &iFrameCount)
{
	cv::Mat tempImg;
	cv::Mat tarImg;

	// 1. Every 50 frames updates the reference image
	if (iFrameCount % 50 == 1)
	{
		m_refImgBuffer->DeQueue(tempImg);
		m_Fftcc2DPtr->ResetRefImg(tempImg);

		// 3.1 Use the results to update the POI positions if iFrameCount is greater than 50
		if (iFrameCount > 50)
		{
			// Update the accumulative current [U,V]
			cudaMemcpy(m_d_fAccumulateU, m_d_fU, sizeof(TW::real_t)*m_iNumPOIs, cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_d_fAccumulateV, m_d_fV, sizeof(TW::real_t)*m_iNumPOIs, cudaMemcpyDeviceToDevice);

			// Use the current [U, V] to update the POI positions
			cuUpdatePOIpos(m_d_fU,
						   m_d_fV,
						   m_iNumberX,
						   m_iNumberY,
						   m_Fftcc2DPtr->g_cuHandle.m_d_iPOIXY);

			int *i = new int;
			cudaMemcpy(i, &m_Fftcc2DPtr->g_cuHandle.m_d_iPOIXY[0], sizeof(int), cudaMemcpyDeviceToHost);
			qDebug()<<*i;
			delete i;
		}

		// 3.2 TODO: Copy the iU, iV to host memory for ICGN

		// qDebug()<<"ref";
	}

	m_tarImgBuffer->DeQueue(tarImg);

	// 2. Do the FFTCC computation and add [U,V] to the accumulative current [U,V]
	// and update the POI positions in the target image
	m_Fftcc2DPtr->cuComputeFFTCC(m_d_fU, m_d_fV, m_d_fZNCC, tarImg);
	cuAccumulatePOI_UV(m_d_fAccumulateU,
				       m_d_fAccumulateV,
					   m_Fftcc2DPtr->g_cuHandle.m_d_iPOIXY,
					   m_iNumPOIs,
					   m_d_fU,
					   m_d_fV,
					   m_d_iCurrentPOIXY);

	int *i = new int;
	cudaMemcpy(i, &m_d_iCurrentPOIXY[0], sizeof(int), cudaMemcpyDeviceToHost);
	qDebug()<<*i;
	delete i;

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

	if (m_sharedResources->sharedTexture != nullptr &&
		m_sharedResources->sharedContext != nullptr &&
		m_sharedResources->sharedProgram != nullptr)
	{
		m_sharedResources->sharedContext->makeCurrent(m_sharedResources->sharedSurface);

		m_sharedResources->sharedProgram->bind();
		m_sharedResources->sharedTexture->bind();

		//m_sharedResources->sharedTexture->setData(0, QOpenGLTexture::Red,QOpenGLTexture::UInt8,texture_data);

		checkCudaErrors(cudaMemcpyToArray(m_sharedResources->cudaImgArray,
			0,
			0,
			m_Fftcc2DPtr->g_cuHandle/*TW::paDIC::g_cuHandle*/.m_d_fTarImg,
			tarImg.cols * tarImg.rows,
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