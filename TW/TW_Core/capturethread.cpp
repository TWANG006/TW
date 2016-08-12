#include "capturethread.h"

#include <QDebug>
#include "TW_MatToQImage.h"

CaptureThread::CaptureThread(ImageBufferPtr refImgBuffer,
						     ImageBufferPtr tarImgBuffer,
							 bool isDropFrameIfBufferFull,
							 int iDeviceNumber,
							 int width,
							 int height,
							 ComputationMode computationMode,
							 QObject *parent)
	: m_isDropFrameIfBufferFull(isDropFrameIfBufferFull)
	, m_refImgBuffer(refImgBuffer)
	, m_tarImgBuffer(tarImgBuffer)
	, m_refImgBufferCPU_ICGN(nullptr)
	, m_tarImgBufferCPU_ICGN(nullptr)
	, m_iDeviceNumber(iDeviceNumber)
	, m_iWidth(width)
	, m_iHeight(height)
	, m_computationMode(computationMode)
	, m_iFrameCount(0)
	, m_isAboutToStop(false)
	, QThread(parent)
{}

CaptureThread::CaptureThread(ImageBufferPtr refImgBuffer,
						     ImageBufferPtr tarImgBuffer,
							 ImageBufferPtr refImgBufferCPU_ICGN,
						     ImageBufferPtr tarImgBufferCPU_ICGN,
							 bool isDropFrameIfBufferFull,
							 int iDeviceNumber,
							 int width,
							 int height,
							 ComputationMode computationMode,
							 QObject *parent)
	: m_isDropFrameIfBufferFull(isDropFrameIfBufferFull)
	, m_refImgBuffer(refImgBuffer)
	, m_tarImgBuffer(tarImgBuffer)
	, m_refImgBufferCPU_ICGN(refImgBufferCPU_ICGN)
	, m_tarImgBufferCPU_ICGN(tarImgBufferCPU_ICGN)
	, m_iDeviceNumber(iDeviceNumber)
	, m_iWidth(width)
	, m_iHeight(height)
	, m_computationMode(computationMode)
	, m_iFrameCount(0)
	, m_isAboutToStop(false)
	, QThread(parent)
{}

CaptureThread::~CaptureThread()
{

}


bool CaptureThread::grabTheFirstRefFrame(cv::Mat &firstFrame)
{
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
		qDebug()<<"[Calculation] Camera is not connected yet!";
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
		//m_cap>>m_grabbedFrame;
		if(!m_cap.grab())
			continue;

		// Retrieve frames
		if(m_cap.retrieve(m_grabbedFrame))
		/*if(!m_grabbedFrame.empty())*/
		{
			m_iFrameCount++;
		
			// Convert the fraemt to grayscal
			m_currentFrame = cv::Mat(m_grabbedFrame.clone());
			cv::cvtColor(m_currentFrame,
						 m_currentFrame,
						 CV_BGR2GRAY);
			m_Qimg = TW::Mat2QImage(m_currentFrame);

			// Every 50 frames the refImg should be updated
			if(m_iFrameCount % 50 == 1)
			{
				// Add the frame to refImgBuffer
				m_refImgBuffer->EnQueue(m_currentFrame, m_isDropFrameIfBufferFull);

				if(ComputationMode::GPUFFTCC_CPUICGN == m_computationMode)
					m_refImgBufferCPU_ICGN->EnQueue(m_currentFrame, m_isDropFrameIfBufferFull);

				// update the GUI's reference image label
				emit newRefQImg(m_Qimg);
			}

			if (m_iFrameCount >= 50 && m_iFrameCount % 50 == 0 && m_computationMode == ComputationMode::GPUFFTCC_CPUICGN)
			{
				m_tarImgBufferCPU_ICGN->EnQueue(m_currentFrame, m_isDropFrameIfBufferFull);
			}
		
			// Add all the loaded frames to the tarImgbuffer
			m_tarImgBuffer->EnQueue(m_currentFrame, m_isDropFrameIfBufferFull);	
			// Update the GUI's target image label
			emit newTarFrame(m_iFrameCount);
			emit newTarQImg(m_Qimg);
		}
	}	

	qDebug()<<"[Calculation] Stopping the Capture thread....";
}

