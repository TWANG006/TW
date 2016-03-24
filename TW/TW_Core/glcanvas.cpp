#include "glcanvas.h"

GLCanvas::GLCanvas(int iDeviceNumber,
				   bool isDropFrameEnabled,
				   int iImgWidth, int iImgHeight,
				   int iSubsetX, int iSubsetY,
				   int iGridSpaceX, int iGridSpaceY,
				   int iMarginX, int iMarginY,
				   ImageBufferPtr refImgBuffer,
				   ImageBufferPtr tarImgBuffer,
				   const QRect& roi,
				   QWidget *parent)
	: m_isCameraConnected(false)
	, m_refImgBuffer(refImgBuffer)
	, m_tarImgBuffer(tarImgBuffer)
	, QOpenGLWidget(parent)
{
	ui.setupUi(this);

	setMinimumSize(640, 480);
	
	/*------------------------Setup the thread connections here-------------------------*/
	/*-1. Create the capture thread-*/
	m_captureThread - new CaptureThread(m_refImgBuffer,
										m_tarImgBuffer,
										isDropFrameEnabled,
										iDeviceNumber,
										iImgWidth,
										iImgHeight,
										this);
	/*-2. Try to connect the camera. If connected retrieve its first frame to-*/
	/*-   fftccWorker. NOTE: this method is executed in the main thread      -*/
	cv::Mat firstFrame;
	m_captureThread->grabTheFirstRefFrame(firstFrame);
	m_fftccWorker = new FFTCCTWorkerThread(m_refImgBuffer,
										   m_tarImgBuffer,
										   iImgWidth, iImgHeight,
										   iSubsetX, iSubsetY,
										   iGridSpaceX, iGridSpaceY,
										   iMarginX, iMarginY,
										   roi,
										   firstFrame);
	/*-3. Move the m_fftccWorker to its own thread m_fftccWorkerThread         -*/
	m_fftccWorkerThread = new QThread;
	m_fftccWorker->moveToThread(m_fftccWorkerThread);
	/*-4. Connect signals/slots between threads                                -*/
	connect(m_fftccWorkerThread, &QThread::finished, m_fftccWorker, &FFTCCTWorkerThread::deleteLater);
	connect(m_fftccWorkerThread, &QThread::finished, m_fftccWorkerThread, &QThread::deleteLater);
	connect(m_captureThread, &CaptureThread::newTarFrame, m_fftccWorker, &FFTCCTWorkerThread::processFrame);
	connect(this, &GLCanvas::renderRequest, m_fftccWorker, &FFTCCTWorkerThread::render);
}

GLCanvas::~GLCanvas()
{

}
