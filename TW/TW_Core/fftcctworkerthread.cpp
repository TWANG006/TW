#include "fftcctworkerthread.h"
#include <QDebug>

#include "cuda_utils.cuh"
#include "TW_MemManager.h"
#include "TW_cuTempUtils.h"
#include <omp.h>

FFTCCTWorkerThread::FFTCCTWorkerThread(
	ImageBufferPtr refImgBuffer,
	ImageBufferPtr tarImgBuffer,
	int iWidth, int iHeight,
	int iSubsetX, int iSubsetY,
	int iGridSpaceX, int iGridSpaceY,
	int iMarginX, int iMarginY,
	const QRect &roi,
	const cv::Mat &firstFrame,
	std::shared_ptr<SharedResources>& s,
	ComputationMode computationMode)
	: m_refImgBuffer(refImgBuffer)
	, m_tarImgBuffer(tarImgBuffer)
	, m_fUBuffer(nullptr)
	, m_fVBuffer(nullptr)
	, m_iPOIXYBuffer(nullptr)
	, m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_ROI(roi)
	, m_Fftcc2DPtr(nullptr)
	, m_Icgn2DPtr(nullptr)
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
	, m_iSubsetX(iSubsetX)
	, m_iSubsetY(iSubsetY)
	, m_averageFPS(0)
	, m_fpsSum(0)
	, m_processingTime(0)
	, m_sampleNumber(0)
	, m_computationMode(computationMode)
{
	// Do the initialization for the paDIC's cuFFTCC here in the constructor
	// 1. Construct the cuFFTCC2D object using the whole image
	m_Fftcc2DPtr.reset(new TW::paDIC::cuFFTCC2D(
		iWidth, iHeight,
		m_ROI.width(), m_ROI.height(),
		m_ROI.x(), m_ROI.y(),
		iSubsetX, iSubsetY,
		iGridSpaceX, iGridSpaceY,
		iMarginX, iMarginY));

	m_iNumberX = m_Fftcc2DPtr->GetNumPOIsX();
	m_iNumberY = m_Fftcc2DPtr->GetNumPOIsY();

	// 1.0 Construct the cuICGN2D object using the whole image
	// NOTE: Currently only parallel Bicubic inerpolation is supported!
	if (m_computationMode == ComputationMode::GPUFFTCC_ICGN)
	{
		m_Icgn2DPtr.reset(new TW::paDIC::cuICGN2D(
		iWidth, iHeight,
		m_ROI.x(), m_ROI.y(),
		m_ROI.width(), m_ROI.height(),
		iSubsetX, iSubsetY,
		m_iNumberX, m_iNumberY,
		20,
		0.001,
		TW::paDIC::ICGN2DInterpolationFLag::Bicubic));
	}

	// Allocate memory for the current U & V
	m_iNumPOIs = m_Fftcc2DPtr->GetNumPOIs();
	cudaMalloc((void**)&m_d_fAccumulateU, sizeof(TW::real_t)*m_iNumPOIs);
	cudaMalloc((void**)&m_d_fAccumulateV, sizeof(TW::real_t)*m_iNumPOIs);
	TW::cuInitialize<TW::real_t>(m_d_fAccumulateU, 0, m_iNumPOIs);
	TW::cuInitialize<TW::real_t>(m_d_fAccumulateV, 0, m_iNumPOIs);

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

	cudaMalloc((void**)&m_d_UColorMap, sizeof(unsigned int)*m_ROI.width()*m_ROI.height());
	cudaMalloc((void**)&m_d_VColorMap, sizeof(unsigned int)*m_ROI.width()*m_ROI.height());

	TW::cuInitialize<unsigned int>(m_d_UColorMap, 0x00FFFFFF, m_ROI.width()*m_ROI.height());
	TW::cuInitialize<unsigned int>(m_d_VColorMap, 0x00FFFFFF, m_ROI.width()*m_ROI.height());
}

FFTCCTWorkerThread::FFTCCTWorkerThread(
	ImageBufferPtr refImgBuffer,
	ImageBufferPtr tarImgBuffer,
	VecBufferfPtr fUBuffer,
	VecBufferfPtr fVBuffer,
	VecBufferiPtr iPOIXYBuffer,
	int iWidth, int iHeight,
	int iSubsetX, int iSubsetY,
	int iGridSpaceX, int iGridSpaceY,
	int iMarginX, int iMarginY,
	const QRect &roi,
	const cv::Mat &firstFrame,
	std::shared_ptr<SharedResources>& s,
	ComputationMode computationMode)
	: m_refImgBuffer(refImgBuffer)
	, m_tarImgBuffer(tarImgBuffer)
	, m_fUBuffer(fUBuffer)
	, m_fVBuffer(fVBuffer)
	, m_iPOIXYBuffer(iPOIXYBuffer)
	, m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_ROI(roi)
	, m_Fftcc2DPtr(nullptr)
	, m_Icgn2DPtr(nullptr)
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
	, m_iSubsetX(iSubsetX)
	, m_iSubsetY(iSubsetY)
	, m_averageFPS(0)
	, m_fpsSum(0)
	, m_processingTime(0)
	, m_sampleNumber(0)
	, m_computationMode(computationMode)
{
	// Do the initialization for the paDIC's cuFFTCC here in the constructor
	// 1. Construct the cuFFTCC2D object using the whole image
	m_Fftcc2DPtr.reset(new TW::paDIC::cuFFTCC2D(
		iWidth, iHeight,
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
	TW::cuInitialize<TW::real_t>(m_d_fAccumulateU, 0, m_iNumPOIs);
	TW::cuInitialize<TW::real_t>(m_d_fAccumulateV, 0, m_iNumPOIs);

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

	cudaMalloc((void**)&m_d_UColorMap, sizeof(unsigned int)*m_ROI.width()*m_ROI.height());
	cudaMalloc((void**)&m_d_VColorMap, sizeof(unsigned int)*m_ROI.width()*m_ROI.height());

	TW::cuInitialize<unsigned int>(m_d_UColorMap, 0x00FFFFFF, m_ROI.width()*m_ROI.height());
	TW::cuInitialize<unsigned int>(m_d_VColorMap, 0x00FFFFFF, m_ROI.width()*m_ROI.height());

	// Allocate memory for the Vectors
	m_h_iPOIXY.resize(m_iNumPOIs * 2);
	m_h_fU.resize(m_iNumPOIs);
	m_h_fV.resize(m_iNumPOIs);
}

FFTCCTWorkerThread::~FFTCCTWorkerThread()
{
	m_Fftcc2DPtr->cuDestroyFFTCC(m_d_fU, m_d_fV, m_d_fZNCC);
	m_Icgn2DPtr->cuFinalize();

	TW::cudaSafeFree(m_d_fAccumulateU);
	TW::cudaSafeFree(m_d_fAccumulateV);
	TW::cudaSafeFree(m_d_fMaxU);
	TW::cudaSafeFree(m_d_fMinU);
	TW::cudaSafeFree(m_d_fMaxV);
	TW::cudaSafeFree(m_d_fMaxV);
	TW::cudaSafeFree(m_d_iCurrentPOIXY);
	TW::cudaSafeFree(m_d_UColorMap);
	TW::cudaSafeFree(m_d_VColorMap);

	cudaDeviceReset();

	deleteObject(m_sharedResources->sharedContext);
}




void FFTCCTWorkerThread::processFrameFFTCC(const int &iFrameCount)
{
	cv::Mat tempImg;
	cv::Mat tarImg;

	m_processingTime = m_t.elapsed();
	m_t.start();

	// 1. Every 50 frames updates the reference image
	if (iFrameCount % 50 == 1)
	{
		m_refImgBuffer->DeQueue(tempImg);
		m_Fftcc2DPtr->ResetRefImg(tempImg);

//		omp_set_num_threads(3);
//				float j = 0;
//#pragma omp parallel
//	{
//#pragma omp for
//		{
//			for (int i = 0; i < 100000000; i++)
//			{
//				j += 0.5;
//			}
//		}
//	}
//	emit testSignal(j);
		// 3.1 Use the results to update the POI positions if iFrameCount is greater than 50
		if (iFrameCount > 50)
		{
			/*cudaMemcpy(m_d_fAccumulateU, m_d_fU, sizeof(TW::real_t)*m_iNumPOIs, cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_d_fAccumulateV, m_d_fV, sizeof(TW::real_t)*m_iNumPOIs, cudaMemcpyDeviceToDevice);*/

			// If CPU-ICGN is the computation mode, copy the iU, iV and POIpos from GPU to CPU for
			// its use. Then, emit the signal ICGNDtaReady to notify ICGN thread to begin processing.
			if(m_computationMode == ComputationMode::GPUFFTCC_CPUICGN)
			{
				cudaMemcpy(m_h_fU.data(), m_d_fU, sizeof(float)*m_iNumPOIs, cudaMemcpyDeviceToHost);
				cudaMemcpy(m_h_fV.data(), m_d_fV, sizeof(float)*m_iNumPOIs, cudaMemcpyDeviceToHost);
				cudaMemcpy(m_h_iPOIXY.data(), m_Fftcc2DPtr->g_cuHandle.m_d_iPOIXY, sizeof(int)*m_iNumPOIs * 2, cudaMemcpyDeviceToHost);

				m_fUBuffer->EnQueue(m_h_fU);
				m_fVBuffer->EnQueue(m_h_fV);
				m_iPOIXYBuffer->EnQueue(m_h_iPOIXY);

				emit ICGNDataReady();
			}

			// Update the accumulative current [U,V]
			cuAccumulateUV(m_d_fAccumulateU, m_d_fAccumulateV, m_iNumPOIs, m_d_fU, m_d_fV);

			// Use the current [U, V] to update the POI positions
			cuUpdatePOIpos(m_d_fU,
				m_d_fV,
				m_iNumberX,
				m_iNumberY,
				m_Fftcc2DPtr->g_cuHandle.m_d_iPOIXY);
		}

		// 3.2 TODO: Copy the iU, iV to host memory for ICGN

		// qDebug()<<"ref";
	}

	m_tarImgBuffer->DeQueue(tarImg);

	// 2. Do the FFTCC computation and add [U,V] to the accumulative current [U,V]
	// and update the POI positions in the target image
	m_Fftcc2DPtr->cuComputeFFTCC(m_d_fU, m_d_fV, m_d_fZNCC, tarImg);


	// 4. Calculate the color map for the iU and iV images
	/*cuAccumulatePOI(m_d_fU,
					m_d_fV,
					m_Fftcc2DPtr->g_cuHandle.m_d_iPOIXY,
					m_iNumPOIs,
					m_d_iCurrentPOIXY);
					minMaxRWrapper(m_d_fU, m_d_fV, m_iNumPOIs, m_iNumPOIs, m_d_fMinU, m_d_fMaxU, m_d_fMinV, m_d_fMaxV);
					constructTextImage(m_d_UColorMap,
					m_d_VColorMap,
					m_d_iCurrentPOIXY,
					m_d_fU,
					m_d_fV,
					m_iNumPOIs,
					m_ROI.x(),
					m_ROI.y(),
					m_ROI.width(),
					m_ROI.height(),
					m_d_fMaxU, m_d_fMinU, m_d_fMaxV, m_d_fMinV);*/
	//cuAccumulateUV(m_d_fU, m_d_fV, m_iNumPOIs, m_d_fAccumulateU, m_d_fAccumulateV);
	cuAccumulatePOI(m_d_fU,
		m_d_fV,
		m_Fftcc2DPtr->g_cuHandle.m_d_iPOIXY,
		m_iNumPOIs,
		m_d_iCurrentPOIXY);
	constructTextImageFixedMinMax(m_d_UColorMap,
		m_d_VColorMap,
		m_d_iCurrentPOIXY,
		m_d_fU,
		m_d_fV,
		m_d_fAccumulateU,
		m_d_fAccumulateV,
		m_iNumPOIs,
		m_ROI.x(),
		m_ROI.y(),
		m_ROI.width(),
		m_ROI.height(),
		20, -20, 20, -20);


	//float *i = new float;
	//cudaMemcpy(i, &m_d_fMinV[0], sizeof(float), cudaMemcpyDeviceToHost);
	//qDebug()<<*i;
	//delete i;


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

		m_sharedResources->sharedProgram->bind();
		m_sharedResources->sharedUTexture->bind();

		checkCudaErrors(cudaMemcpyToArray(m_sharedResources->cudaUArray,
			0,
			0,
			m_d_UColorMap,
			sizeof(unsigned int)*m_ROI.width()*m_ROI.height(),
			cudaMemcpyDeviceToDevice));

		m_sharedResources->sharedUTexture->release();
		m_sharedResources->sharedProgram->release();

		m_sharedResources->sharedProgram->bind();
		m_sharedResources->sharedVTexture->bind();

		checkCudaErrors(cudaMemcpyToArray(m_sharedResources->cudaVArray,
			0,
			0,
			m_d_VColorMap,
			sizeof(unsigned int)*m_ROI.width()*m_ROI.height(),
			cudaMemcpyDeviceToDevice));

		m_sharedResources->sharedVTexture->release();
		m_sharedResources->sharedProgram->release();

		m_sharedResources->sharedContext->doneCurrent();

		emit frameReady();
		TW::cuInitialize<unsigned int>(m_d_UColorMap, 0x00FFFFFF, m_ROI.width()*m_ROI.height());
		TW::cuInitialize<unsigned int>(m_d_VColorMap, 0x00FFFFFF, m_ROI.width()*m_ROI.height());
	}

	updateFPS(m_processingTime);
	emit runningStaticsReady(m_iNumPOIs, m_averageFPS);
	/*Deperacated*/
	// 5. Invoke the CUDA & OpenGL interoperability
	// 5.1 Map the target image data
	// 5.2 Map the colormap data
	// 5.3 Normalize to [0,1] scale
	// delete i; i = nullptr;
}

void FFTCCTWorkerThread::processFrameFFTCC_ICGN(const int &iFrameCount)
{
	cv::Mat tempImg;
	cv::Mat tarImg;

	m_processingTime = m_t.elapsed();
	m_t.start();

	// 1. Every 50 frames updates the reference image
	if (iFrameCount % 50 == 1)
	{
		m_refImgBuffer->DeQueue(tempImg);
		m_Fftcc2DPtr->ResetRefImg(tempImg);
		
		// Initialize the refImg for the ICGN
		m_Icgn2DPtr->cuInitialize(m_Fftcc2DPtr->g_cuHandle.m_d_fRefImg);

		// 3.1 Use the results to update the POI positions if iFrameCount is greater than 50
		if (iFrameCount > 50)
		{
			// Update the accumulative current [U,V]
			/*cudaMemcpy(m_d_fAccumulateU, m_d_fU, sizeof(TW::real_t)*m_iNumPOIs, cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_d_fAccumulateV, m_d_fV, sizeof(TW::real_t)*m_iNumPOIs, cudaMemcpyDeviceToDevice);*/


			cuAccumulateUV(m_d_fAccumulateU, m_d_fAccumulateV, m_iNumPOIs, m_d_fU, m_d_fV);

			// Use the current [U, V] to update the POI positions
			cuUpdatePOIpos(
				m_d_fU,
				m_d_fV,
				m_iNumberX,
				m_iNumberY,
				m_Fftcc2DPtr->g_cuHandle.m_d_iPOIXY);

			//int *i = new int;
			//cudaMemcpy(i, &m_Fftcc2DPtr->g_cuHandle.m_d_iPOIXY[0], sizeof(int), cudaMemcpyDeviceToHost);
			//qDebug()<<*i;
			//delete i;
		}

		// 3.2 TODO: Copy the iU, iV to host memory for ICGN

		// qDebug()<<"ref";
	}

	m_tarImgBuffer->DeQueue(tarImg);

	// 2. Do the FFTCC computation and add [U,V] to the accumulative current [U,V]
	// and update the POI positions in the target image
	m_Fftcc2DPtr->cuComputeFFTCC(
		m_d_fU, 
		m_d_fV,
		m_d_fZNCC,
		tarImg);

	// For debug

	m_Icgn2DPtr->cuCompute(
		m_Fftcc2DPtr->g_cuHandle.m_d_fTarImg,
		m_Fftcc2DPtr->g_cuHandle.m_d_iPOIXY,
		m_d_fU,
		m_d_fV);

	// 4. Calculate the color map for the iU and iV images
	/*cuAccumulatePOI(m_d_fU,
					m_d_fV,
					m_Fftcc2DPtr->g_cuHandle.m_d_iPOIXY,
					m_iNumPOIs,
					m_d_iCurrentPOIXY);
					minMaxRWrapper(m_d_fU, m_d_fV, m_iNumPOIs, m_iNumPOIs, m_d_fMinU, m_d_fMaxU, m_d_fMinV, m_d_fMaxV);
					constructTextImage(m_d_UColorMap,
					m_d_VColorMap,
					m_d_iCurrentPOIXY,
					m_d_fU,
					m_d_fV,
					m_iNumPOIs,
					m_ROI.x(),
					m_ROI.y(),
					m_ROI.width(),
					m_ROI.height(),
					m_d_fMaxU, m_d_fMinU, m_d_fMaxV, m_d_fMinV);*/
	//cuAccumulateUV(m_d_fU, m_d_fV, m_iNumPOIs, m_d_fAccumulateU, m_d_fAccumulateV);
	cuAccumulatePOI(
		m_d_fU,
		m_d_fV,
		m_Fftcc2DPtr->g_cuHandle.m_d_iPOIXY,
		m_iNumPOIs,
		m_d_iCurrentPOIXY);
	constructTextImageFixedMinMax(
		m_d_UColorMap,
		m_d_VColorMap,
		m_d_iCurrentPOIXY,
		m_d_fU,
		m_d_fV,
		m_d_fAccumulateU,
		m_d_fAccumulateV,
		m_iNumPOIs,
		m_ROI.x(),
		m_ROI.y(),
		m_ROI.width(),
		m_ROI.height(),
		20, -20, 20, -20);


	//float *i = new float;
	//cudaMemcpy(i, &m_d_fMinV[0], sizeof(float), cudaMemcpyDeviceToHost);
	//qDebug()<<*i;
	//delete i;


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

		m_sharedResources->sharedProgram->bind();
		m_sharedResources->sharedUTexture->bind();

		checkCudaErrors(cudaMemcpyToArray(m_sharedResources->cudaUArray,
			0,
			0,
			m_d_UColorMap,
			sizeof(unsigned int)*m_ROI.width()*m_ROI.height(),
			cudaMemcpyDeviceToDevice));

		m_sharedResources->sharedUTexture->release();
		m_sharedResources->sharedProgram->release();

		m_sharedResources->sharedProgram->bind();
		m_sharedResources->sharedVTexture->bind();

		checkCudaErrors(cudaMemcpyToArray(m_sharedResources->cudaVArray,
			0,
			0,
			m_d_VColorMap,
			sizeof(unsigned int)*m_ROI.width()*m_ROI.height(),
			cudaMemcpyDeviceToDevice));

		m_sharedResources->sharedVTexture->release();
		m_sharedResources->sharedProgram->release();

		m_sharedResources->sharedContext->doneCurrent();

		emit frameReady();
		TW::cuInitialize<unsigned int>(m_d_UColorMap, 0x00FFFFFF, m_ROI.width()*m_ROI.height());
		TW::cuInitialize<unsigned int>(m_d_VColorMap, 0x00FFFFFF, m_ROI.width()*m_ROI.height());
	}

	updateFPS(m_processingTime);
	emit runningStaticsReady(m_iNumPOIs, m_averageFPS);
	/*Deperacated*/
	// 5. Invoke the CUDA & OpenGL interoperability
	// 5.1 Map the target image data
	// 5.2 Map the colormap data
	// 5.3 Normalize to [0,1] scale
	// delete i; i = nullptr;
}

void FFTCCTWorkerThread::updateFPS(int timeElapsed)
{
	// Add instantaneous FPS value to queue
	if (timeElapsed > 0)
	{
		m_fps.enqueue((int)1000 / timeElapsed);
		// Increment sample number
		m_sampleNumber++;
	}

	// Maximum size of queue is DEFAULT_PROCESSING_FPS_STAT_QUEUE_LENGTH
	if (m_fps.size() > PROCESSING_FPS_STAT_QUEUE_LENGTH)
	{
		m_fps.dequeue();
	}

	// Update FPS value every DEFAULT_PROCESSING_FPS_STAT_QUEUE_LENGTH samples
	if ((m_fps.size() == PROCESSING_FPS_STAT_QUEUE_LENGTH) && (m_sampleNumber == PROCESSING_FPS_STAT_QUEUE_LENGTH))
	{
		// Empty queue and store sum
		while (!m_fps.empty())
		{
			m_fpsSum += m_fps.dequeue();
		}
		// Calculate average FPS
		m_averageFPS = m_fpsSum / PROCESSING_FPS_STAT_QUEUE_LENGTH;
		// Reset sum
		m_fpsSum = 0;
		// Reset sample number
		m_sampleNumber = 0;
	}
}
