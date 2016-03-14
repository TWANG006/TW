#include "capturethread.h"

#include <QDebug>

CaptureThread::CaptureThread(ImageBufferPtr refImgBuffer,
						     ImageBufferPtr tarImgBuffer,
							 bool isDropFrameIfBufferFull,
							 int iDeviceNumber,
							 int width,
							 int height,
							 QObject *parent)
	: m_isDropFrameIfBufferFull(isDropFrameIfBufferFull)
	, m_refImgBuffer(refImgBuffer)
	, m_tarImgBuffer(tarImgBuffer)
	, m_iDeviceNumber(iDeviceNumber)
	, m_iWidth(width)
	, m_iHeight(height)
	, m_iFrameCount(0)
	, m_isAboutToStop(false)
	, QThread(parent)
{}

CaptureThread::~CaptureThread()
{

}


bool CaptureThread::grabTheFirstRefFrame(cv::Mat &firstFrame)
{
	if(!m_cap.open(m_iDeviceNumber))
		return false;
	if(!m_cap.grab())
		return false;
	if(!m_cap.retrieve(firstFrame))
		return false;

	return true;
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
	forever
	{
		// Check whether the stop thread is triggered or not
		m_stopMutex.lock();
		if(m_isAboutToStop)
		{
			m_isAboutToStop = false;
			m_stopMutex.unlock();
			break;
		}
		m_stopMutex.unlock();

	
		// Retrieve frame
		if(!m_cap.grab())
			continue;

		// Retrieve frames
		m_cap.retrieve(m_grabbedFrame);
		m_iFrameCount++;
		
		// Convert the fraemt to grayscal
		m_currentFrame = cv::Mat(m_grabbedFrame.clone());
		cv::cvtColor(m_currentFrame,
					 m_currentFrame,
				     CV_BGR2GRAY);

		// Every 50 frames the refImg should be updated
		if(m_iFrameCount % 50 == 1)
		{
			// Add the frame to refImgBuffer
			m_refImgBuffer->EnQueue(m_currentFrame, m_isDropFrameIfBufferFull);
			
		}		
		
		// Add all the loaded frames to the tarImgbuffer
		m_tarImgBuffer->EnQueue(m_currentFrame, m_isDropFrameIfBufferFull);
				
		emit newTarFrame(m_iFrameCount);
	}	

	qDebug()<<"Stopping the Capture thread....";
}

