#include "onecamwidget.h"
#include <QDebug>
#include <QMessageBox>

OneCamWidget::OneCamWidget(int deviceNumber,
	ImageBufferPtr refImgBuffer,
	ImageBufferPtr tarImgBuffer,
	int iImgWidth,
	int iImgHeight,
	int iNumberX,
	int iNumberY,
	const QRect& roi,
	ComputationMode computationMode,
	QWidget *parent)
	: m_iDeviceNumber(deviceNumber)
	, m_refImgBuffer(refImgBuffer)
	, m_tarImgBuffer(tarImgBuffer)
	, m_fUBuffer(nullptr)
	, m_fVBuffer(nullptr)
	, m_iPOIXYBuffer(nullptr)
	, m_refImgBufferCPU_ICGN(nullptr)
	, m_tarImgBufferCPU_ICGN(nullptr)
	, m_isCameraConnected(false)
	, m_captureThread(nullptr)
	, m_fftccWorker(nullptr)
	, m_fftccWorkerThread(nullptr)
	, m_icgnWorker(nullptr)
	, m_icgnWorkerThread(nullptr)
	, m_iImgWidth(iImgWidth)
	, m_iImgHeight(iImgHeight)
	, m_iNumberX(iNumberX)
	, m_iNumberY(iNumberY)
	, m_computationMode(computationMode)
	, QWidget(parent)
{
	ui.setupUi(this);

	// 1. Create the FFT-CC worker thread
	m_fftccWorkerThread = new QThread;

	//// 1.0 TODO: If needed, create another ICGN worker object
	//if (m_computationMode == ComputationMode::GPUFFTCC_CPUICGN)
	//{
	//	;
	//}

	m_sharedResources.reset(new SharedResources);
	m_twGLwidget = new GLWidget(m_sharedResources,
		m_fftccWorkerThread,
		this,
		m_iImgWidth,
		m_iImgHeight,
		roi);
	ui.gridLayout->addWidget(m_twGLwidget, 0, 1, 1, 1);
}

OneCamWidget::OneCamWidget(
	int deviceNumber,
	ImageBufferPtr refImgBuffer,
	ImageBufferPtr tarImgBuffer,
	ImageBufferPtr refImgBufferCPU_ICGN,
	ImageBufferPtr tarImgBufferCPU_ICGN,
	int iImgWidth,
	int iImgHeight,
	int iNumberX,
	int iNumberY,
	const QRect& roi,
	ComputationMode computationMode,
	QWidget *parent)
	: m_iDeviceNumber(deviceNumber)
	, m_refImgBuffer(refImgBuffer)
	, m_tarImgBuffer(tarImgBuffer)
	, m_refImgBufferCPU_ICGN(refImgBufferCPU_ICGN)
	, m_tarImgBufferCPU_ICGN(tarImgBufferCPU_ICGN)
	, m_fUBuffer(new TW::Concurrent_Buffer<std::vector<float>>(20))
	, m_fVBuffer(new TW::Concurrent_Buffer<std::vector<float>>(20))
	, m_iPOIXYBuffer(new TW::Concurrent_Buffer<std::vector<int>>(20))
	, m_isCameraConnected(false)
	, m_captureThread(nullptr)
	, m_fftccWorker(nullptr)
	, m_fftccWorkerThread(nullptr)
	, m_icgnWorker(nullptr)
	, m_icgnWorkerThread(nullptr)
	, m_iImgWidth(iImgWidth)
	, m_iImgHeight(iImgHeight)
	, m_iNumberX(iNumberX)
	, m_iNumberY(iNumberY)
	, m_computationMode(computationMode)
	, QWidget(parent)
{
	ui.setupUi(this);

	// 1. Create the FFT-CC worker thread
	m_fftccWorkerThread = new QThread;

	// 2. Create the ICGN worker thread
	m_icgnWorkerThread = new QThread;

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


	// 1. Create the capture thread
	if(ComputationMode::GPUFFTCC_CPUICGN == m_computationMode)
	{
		m_captureThread = new CaptureThread(
		m_refImgBuffer,
		m_tarImgBuffer,
		m_refImgBufferCPU_ICGN,
		m_tarImgBufferCPU_ICGN,
		ifDropFrame,
		m_iDeviceNumber,
		m_iImgWidth,
		m_iImgHeight,
		m_computationMode,
		this);
	}
	else
	{
		m_captureThread = new CaptureThread(
		m_refImgBuffer,
		m_tarImgBuffer,
		ifDropFrame,
		m_iDeviceNumber,
		m_iImgWidth,
		m_iImgHeight,
		m_computationMode,
		this);
	}

	connect(m_captureThread, &CaptureThread::finished, m_captureThread, &CaptureThread::deleteLater);

	// 2. Attempt to connect to camera
	if (m_captureThread->connectToCamera())
	{
		// 3. Aquire the first frame to initialize the FFTCCWorker
		cv::Mat firstFrame;
		m_captureThread->grabTheFirstRefFrame(firstFrame);

		// Construct & initialize the fftccWorker; If needed, create the ICGNWorkerThread()
		if(ComputationMode::GPUFFTCC_CPUICGN == m_computationMode)
		{
			m_fftccWorker = new FFTCCTWorkerThread(
				m_refImgBuffer,
				m_tarImgBuffer,
				m_fUBuffer,
				m_fVBuffer,
				m_iPOIXYBuffer,
				m_iImgWidth, m_iImgHeight,
				iSubsetX, iSubsetY,
				iGridSpaceX, iGridSpaceY,
				iMarginX, iMarginY,
				roi,
				firstFrame,
				m_sharedResources,
				m_computationMode);

			m_icgnWorker = new ICGNWorkerThread(
				m_refImgBufferCPU_ICGN,
				m_tarImgBufferCPU_ICGN,
				m_fUBuffer,
				m_fVBuffer,
				m_iPOIXYBuffer,
				roi,
				m_iImgWidth, m_iImgHeight,
				m_iNumberX, m_iNumberY,
				iSubsetX, iSubsetY,
				20,
				0.001);
			m_icgnWorker->moveToThread(m_icgnWorkerThread);
		}
		else
		{
			m_fftccWorker = new FFTCCTWorkerThread(
				m_refImgBuffer,
				m_tarImgBuffer,
				m_iImgWidth, m_iImgHeight,
				iSubsetX, iSubsetY,
				iGridSpaceX, iGridSpaceY,
				iMarginX, iMarginY,
				roi,
				firstFrame,
				m_sharedResources,
				m_computationMode);
		}

		// Move the fftccworker to its own thread		
		m_fftccWorker->moveToThread(m_fftccWorkerThread);
		
		// 6. Do the signal/slot connections here
		connect(m_captureThread, &CaptureThread::newRefQImg, this, &OneCamWidget::updateRefFrame);
		connect(m_captureThread, &CaptureThread::newTarQImg, this, &OneCamWidget::updateTarFrame);
		connect(m_fftccWorker, &FFTCCTWorkerThread::runningStaticsReady, this, &OneCamWidget::updateStatics);
		connect(m_fftccWorkerThread, &QThread::finished, m_fftccWorker, &QObject::deleteLater);
		connect(m_fftccWorkerThread, &QThread::finished, m_fftccWorkerThread, &QThread::deleteLater);
		connect(m_fftccWorker, SIGNAL(frameReady()), m_twGLwidget, SLOT(update()));
		

		// 6.0 Hand-shaking protocol between ICGNThread and QThread
		if (m_computationMode == ComputationMode::GPUFFTCC || m_computationMode == ComputationMode::GPUFFTCC_CPUICGN)
			connect(m_captureThread, &CaptureThread::newTarFrame, m_fftccWorker, &FFTCCTWorkerThread::processFrameFFTCC);
		if (m_computationMode == ComputationMode::GPUFFTCC_CPUICGN)
		{
			// connect the signals for CPU ICGN processes
			connect(m_fftccWorker, &FFTCCTWorkerThread::ICGNDataReady, m_icgnWorker, &ICGNWorkerThread::processFrame);
			connect(m_icgnWorkerThread, &QThread::finished, m_icgnWorker, &QObject::deleteLater);
			connect(m_icgnWorkerThread, &QThread::finished, m_icgnWorkerThread, &QThread::deleteLater);
			connect(m_icgnWorker, &ICGNWorkerThread::testSignal, this, &OneCamWidget::testSlot);
		}
		if (m_computationMode == ComputationMode::GPUFFTCC_ICGN)
			connect(m_captureThread, &CaptureThread::newTarFrame, m_fftccWorker, &FFTCCTWorkerThread::processFrameFFTCC_ICGN);

		// 7. Start the capture & worker threads
		m_captureThread->start();
		m_fftccWorkerThread->start();
		if (m_computationMode == ComputationMode::GPUFFTCC_CPUICGN)
			m_icgnWorkerThread->start();


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

void OneCamWidget::stopICGNWorkerThread()
{
	qDebug() << "[Calculation] [" << m_iDeviceNumber << "] About to stop ICGNWorker thread...";
	if (m_icgnWorkerThread->isRunning())
	{
		m_icgnWorkerThread->quit();
		m_icgnWorkerThread->wait();
	}

	qDebug() << "[Calculation] [" << m_iDeviceNumber << "] ICGNWorker thread successfully stopped...";
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

void OneCamWidget::testSlot(const float &i)
{
	QMessageBox::warning(this, QString::number(i), QString("kanzhe"));
}