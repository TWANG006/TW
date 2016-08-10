#include "onecamwidget.h"
#include <QDebug>

OneCamWidget::OneCamWidget(int deviceNumber,
	ImageBufferPtr refImgBuffer,
	ImageBufferPtr tarImgBuffer,
	int iImgWidth,
	int iImgHeight,
	const QRect& roi,
	ComputationMode computationMode,
	QWidget *parent)
	: m_iDeviceNumber(deviceNumber)
	, m_refImgBuffer(refImgBuffer)
	, m_tarImgBuffer(tarImgBuffer)
	, m_isCameraConnected(false)
	, m_captureThread(nullptr)
	, m_fftccWorker(nullptr)
	, m_fftccWorkerThread(nullptr)
	, m_icgnWorker(nullptr)
	, m_icgnWorkerThread(nullptr)
	, m_iImgWidth(iImgWidth)
	, m_iImgHeight(iImgHeight)
	, m_computationMode(computationMode)
	, QWidget(parent)
{
	ui.setupUi(this);

	// 1. Create the FFT-CC worker thread
	m_fftccWorkerThread = new QThread;

	// 1.0 TODO: If needed, create another ICGN worker object
	if (m_computationMode == ComputationMode::GPUFFTCC_CPUICGN)
	{
		;
	}

	m_sharedResources.reset(new SharedResources);
	m_twGLwidget = new GLWidget(m_sharedResources,
		m_fftccWorkerThread,
		this,
		m_iImgWidth,
		m_iImgHeight,
		roi);
	ui.gridLayout->addWidget(m_twGLwidget, 0, 1, 1, 1);
}

OneCamWidget::~OneCamWidget()
{
	if (m_isCameraConnected)
	{
		//// Stop the fftccWorkerThread
		//if(m_fftccWorkerThread->isRunning())
		//{
		//	stopFFTCCWorkerThread();
		//}

		// Stop the captureThread
		if (m_captureThread->isRunning())
		{
			stopCaptureThread();
		}

		if (m_captureThread->disconnectCamera())
		{
			qDebug() << "[Calculation] [" << m_iDeviceNumber << "] Camera Successfully disconnected.";
		}
		else
		{
			qDebug() << "[Calculation] [" << m_iDeviceNumber << "] WARNING: Camera already disconnected.";
		}
	}
}

bool OneCamWidget::connectToCamera(bool ifDropFrame,
	int iSubsetX, int iSubsetY,
	int iGridSpaceX, int iGridSpaceY,
	int iMarginX, int iMarginY,
	const QRect& roi)
{
	// Update the text of the labels
	ui.refFramelabel->setText(tr("Connecting to camera..."));
	ui.tarFramelabel->setText(tr("Connecting to camera..."));


	// 1. Create the capture thread & the FFTCC worker and its thread
	m_captureThread = new CaptureThread(
		m_refImgBuffer,
		m_tarImgBuffer,
		ifDropFrame,
		m_iDeviceNumber,
		m_iImgWidth,
		m_iImgHeight,
		this);

	connect(m_captureThread, &CaptureThread::finished, m_captureThread, &CaptureThread::deleteLater);

	// 2. Attempt to connect to camera
	if (m_captureThread->connectToCamera())
	{
		// 3. Aquire the first frame to initialize the FFTCCWorker
		cv::Mat firstFrame;
		m_captureThread->grabTheFirstRefFrame(firstFrame);

		// 4. Construct & initialize the fftccWorker
		m_fftccWorker = new FFTCCTWorkerThread(m_refImgBuffer,
			m_tarImgBuffer,
			m_iImgWidth, m_iImgHeight,
			iSubsetX, iSubsetY,
			iGridSpaceX, iGridSpaceY,
			iMarginX, iMarginY,
			roi,
			firstFrame,
			m_sharedResources,
			m_computationMode);

		// 4.0 TODO: If needed, create the ICGNWorkerThread()

		// 5. Move the fftccworker to its own thread		
		m_fftccWorker->moveToThread(m_fftccWorkerThread);

		// 6. Do the signal/slot connections here
		connect(m_captureThread, &CaptureThread::newRefQImg, this, &OneCamWidget::updateRefFrame);
		connect(m_captureThread, &CaptureThread::newTarQImg, this, &OneCamWidget::updateTarFrame);
		connect(m_fftccWorker, &FFTCCTWorkerThread::runningStaticsReady, this, &OneCamWidget::updateStatics);
		connect(m_fftccWorkerThread, &QThread::finished, m_fftccWorker, &QObject::deleteLater);
		connect(m_fftccWorkerThread, &QThread::finished, m_fftccWorkerThread, &QThread::deleteLater);
		if (m_computationMode == ComputationMode::GPUFFTCC || m_computationMode == ComputationMode::GPUFFTCC_CPUICGN)
			connect(m_captureThread, &CaptureThread::newTarFrame, m_fftccWorker, &FFTCCTWorkerThread::processFrameFFTCC);
		if (m_computationMode == ComputationMode::GPUFFTCC_ICGN)
			connect(m_captureThread, &CaptureThread::newTarFrame, m_fftccWorker, &FFTCCTWorkerThread::processFrameFFTCC_ICGN);
		connect(m_fftccWorker, SIGNAL(frameReady()), m_twGLwidget, SLOT(update()));

		// 6.0 TODO: Hand-shaking protocol between ICGNThread and QThread

		// 7. Start the capture & worker threads
		m_captureThread->start();
		m_fftccWorkerThread->start();



		m_isCameraConnected = true;
		return true;
	}
	else
	{
		return false;
	}
}

void OneCamWidget::stopCaptureThread()
{
	qDebug() << "[Calculation] [" << m_iDeviceNumber << "] About to stop capture thread...";
	m_captureThread->stop();

	// In case the thread is in waiting state
	if (m_refImgBuffer->IsFull())
		m_refImgBuffer->DeQueue();
	if (m_tarImgBuffer->IsFull())
		m_tarImgBuffer->DeQueue();

	m_captureThread->wait();

	qDebug() << "[Calculation] [" << m_iDeviceNumber << "] Capture thread successfully stopped.";
}

void OneCamWidget::stopFFTCCWorkerThread()
{
	qDebug() << "[Calculation] [" << m_iDeviceNumber << "] About to stop FFTCCWorker thread...";
	if (m_fftccWorkerThread->isRunning())
	{
		m_fftccWorkerThread->quit();
		m_fftccWorkerThread->wait();
	}

	qDebug() << "[Calculation] [" << m_iDeviceNumber << "] FFTCCWorker thread successfully stopped...";
}

void OneCamWidget::updateRefFrame(const QImage& refImg)
{
	ui.refFramelabel->setPixmap(QPixmap::fromImage(refImg).scaled(ui.refFramelabel->width(),
		ui.refFramelabel->height(),
		Qt::KeepAspectRatio));
}

void OneCamWidget::updateTarFrame(const QImage& tarImg)
{
	ui.tarFramelabel->setPixmap(QPixmap::fromImage(tarImg).scaled(ui.tarFramelabel->width(),
		ui.tarFramelabel->height(),
		Qt::KeepAspectRatio));
}

void OneCamWidget::updateStatics(const int& iNumPOI, const int& iFPS)
{
	QString qstr = QLatin1String("Number of POIs is: ") + QString::number(iNumPOI)
		+ QLatin1String("    FPS = ") + QString::number(iFPS);
	emit titleReady(qstr);
}