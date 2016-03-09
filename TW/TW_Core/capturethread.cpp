#include "capturethread.h"

#include <QDebug>

CaptureThread::CaptureThread(ImageBufferPtr imgBuffer,
							 bool isDropFrameIfBufferFull,
							 int iDeviceNumber,
							 int width,
							 int height,
							 QObject *parent)
	: m_isDropFrameIfBufferFull(isDropFrameIfBufferFull)
	, m_imgBuffer(imgBuffer)
	, m_iDeviceNumber(iDeviceNumber)
	, m_iWidth(width)
	, m_iHeight(height)
	, m_isAboutToStop(false)
	, QThread(parent)
{}

CaptureThread::~CaptureThread()
{

}

bool CaptureThread::connectToCamera()
{
	bool isCamOpened = m_cap.open(m_iDeviceNumber);

	if(m_iWidth != -1)
		m_cap.set(CV_CAP_PROP_FRAME_WIDTH, m_iWidth);
	if(m_iHeight)
		m_cap.set(CV_CAP_PROP_FRAME_HEIGHT, m_iHeight);

	return isCamOpened;
}

bool CaptureThread::disconnectCamera()
{
	if(m_cap.isOpened())
	{
		m_cap.release();
		return true;
	}

	else
	{
		qDebug()<<"Camera is not connected yet!";
		return false;
	}
}

void CaptureThread::stop()
{
	QMutexLocker locker(&m_stopMutex);
	m_isAboutToStop = true;
}

void CaptureThread::run()
{

	qDebug()<<"Stopping the Capture thread....";
}

