#include "fftcc1camwidget.h"
#include <QDebug>

FFTCC1CamWidget::FFTCC1CamWidget(int deviceNumber,
								 ImageBufferPtr refImgBuffer,
								 ImageBufferPtr tarImgBuffer,
								 QWidget *parent)
	: m_iDeviceNumber(deviceNumber)
	, m_refImgBuffer(refImgBuffer)
	, m_tarImgBuffer(tarImgBuffer)
	, m_isCameraConnected(false)
	, m_captureThread(nullptr)
	, m_fftccWorker(nullptr)
	, QWidget(parent)
{
	ui.setupUi(this);
}

FFTCC1CamWidget::~FFTCC1CamWidget()
{
	if(m_isCameraConnected)
	{
		// Stop the fftccWorkerThread
		if(m_fftccWorkerThread.isRunning())
		{
			stopFFTCCWorkerThread();
		}

		// Stop the captureThread
		if(m_captureThread->isRunning())
		{
			stopCaptureThread();
		}

		if(m_captureThread->disconnectCamera())
		{
			qDebug() <<"["<<m_iDeviceNumber<<"] Camera Successfully disconnected.";
		}
		else
		{
			qDebug() <<"["<<m_iDeviceNumber<<"] WARNING: Camera already disconnected.";
		}
	}
}

bool FFTCC1CamWidget::connectToCamera(bool ifDropFrame, 
									  int width, int height,
									  int iSubsetX, int iSubsetY,
									  int iGridSpaceX, int iGridSpaceY,
									  int iMarginX, int iMarginY,
									  const QRect& roi)
{
	// Update the text of the labels
	ui.refFramelabel->setText(tr("Connecting to camera..."));
	ui.tarFramelabel->setText(tr("Connecting to camera..."));

	// Create the capture thread & the FFTCC worker and its thread
	m_captureThread = new CaptureThread(m_refImgBuffer,
										m_tarImgBuffer,
										ifDropFrame,
										m_iDeviceNumber,
										width,
										height,
										this);

	// Attempt to connect to camera
	if(m_captureThread->connectToCamera())
	{
		// Aquire the first frame to initialize the FFTCCWorker
		cv::Mat firstFrame;
		m_captureThread->grabTheFirstRefFrame(firstFrame);

		// Construct the fftccWorker
		m_fftccWorker = new FFTCCTWorkerThread(m_refImgBuffer,
											   m_tarImgBuffer,
											   width, height,
											   iSubsetX, iSubsetY,
											   iGridSpaceX, iGridSpaceY,
											   iMarginX, iMarginY,
											   roi,
											   firstFrame);
		// Move the fftccworker to its own thread
		m_fftccWorker->moveToThread(&m_fftccWorkerThread);

		// Do the signal/slot connections here
		connect(m_captureThread, &CaptureThread::newRefQImg, this, &FFTCC1CamWidget::updateRefFrame);
		connect(m_captureThread, &CaptureThread::newTarQImg, this, &FFTCC1CamWidget::updateTarFrame);
		connect(&m_fftccWorkerThread, &QThread::finished, m_fftccWorker.data(), &QObject::deleteLater);
		connect(&m_fftccWorkerThread, &QThread::finished, &m_fftccWorkerThread, &QThread::deleteLater);
		connect(m_captureThread, &CaptureThread::newTarFrame, m_fftccWorker.data(), &FFTCCTWorkerThread::processFrame);

		m_captureThread->start();
		m_fftccWorkerThread.start();

		m_isCameraConnected = true;
		
		return true;
	}

	else
	{
		return false;
	}
}

void FFTCC1CamWidget::stopCaptureThread()
{
	qDebug() << "["<<m_iDeviceNumber<<"] About to stop capture thread...";
	m_captureThread->stop();

	// In case the thread is in waiting state
	if(m_refImgBuffer->IsFull())
		m_refImgBuffer->DeQueue();
	if(m_tarImgBuffer->IsFull())
		m_tarImgBuffer->DeQueue();

	m_captureThread->wait();

	qDebug() <<"["<<m_iDeviceNumber<<"] Capture thread successfully stopped.";
}

void FFTCC1CamWidget::stopFFTCCWorkerThread()
{
	qDebug() << "["<<m_iDeviceNumber<<"] About to stop FFTCCWorker thread...";
	if(m_fftccWorkerThread.isRunning())
	{
		m_fftccWorkerThread.quit();
		m_fftccWorkerThread.wait();
	}

	qDebug() <<"["<<m_iDeviceNumber<<"] FFTCCWorker thread successfully stopped...";
}

void FFTCC1CamWidget::updateRefFrame(const QImage& refImg)
{
	ui.refFramelabel->setPixmap(QPixmap::fromImage(refImg).scaled(ui.refFramelabel->width(), 
															      ui.refFramelabel->height(), 
															      Qt::KeepAspectRatio));
}

void FFTCC1CamWidget::updateTarFrame(const QImage& tarImg)
{
	ui.tarFramelabel->setPixmap(QPixmap::fromImage(tarImg).scaled(ui.tarFramelabel->width(), 
															      ui.tarFramelabel->height(), 
															      Qt::KeepAspectRatio));
}