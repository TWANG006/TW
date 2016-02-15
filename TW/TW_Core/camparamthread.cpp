#include "camparamthread.h"

CamParamThread::CamParamThread(int width, 
							   int height)
	: m_width(width)
	, m_height(height)
{
	
}

CamParamThread::~CamParamThread()
{

}

bool CamParamThread::connectToCamera()
{
	// Open camera
	return (m_cap.open(0));
}

bool CamParamThread::disconnectCamera()
{
	// Camera is opened
	if(m_cap.isOpened())
	{
		// Disconnect camera
		m_cap.release();
		return true;
	}

	else
	{
		return false;
	}
}

void CamParamThread::stop()
{
	QMutexLocker locker(&m_mutex);
	m_doStop = true;
}

bool CamParamThread::isCameraConnected()
{
    return m_cap.isOpened();
}

int CamParamThread::getInputSourceWidth()
{
    return m_cap.get(CV_CAP_PROP_FRAME_WIDTH);
}

int CamParamThread::getInputSourceHeight()
{
    return m_cap.get(CV_CAP_PROP_FRAME_HEIGHT);
}


void CamParamThread::setROI(QRect roi)
{

}

void CamParamThread::run()
{

}