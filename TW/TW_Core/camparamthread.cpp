#include "camparamthread.h"

#include "TW_MatToQImage.h"

CamParamThread::CamParamThread(int width, 
							   int height)
	: m_width(width)
	, m_height(height)
	, m_doStop(false)
{
	
}

CamParamThread::~CamParamThread()
{
	
}

bool CamParamThread::connectToCamera()
{
	 // Open camera
    bool camOpenResult = m_cap.open(0);
    // Set resolution
    if (m_width != -1)
    {
        m_cap.set(CV_CAP_PROP_FRAME_WIDTH, m_width);
    }
    if (m_height != -1)
    {
        m_cap.set(CV_CAP_PROP_FRAME_HEIGHT, m_height);
    }
    // Return result
    return camOpenResult;
}

bool CamParamThread::disconnectCamera()
{
	// Camera is connected
    if (m_cap.isOpened())
    {
		// Disconnect camera
        m_cap.release();
        return true;
    }
	
	// Camera is NOT connected
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
	while(1)
	{
		// Stop the thread if doStop = true
		m_mutex.lock();
		if(m_doStop)
		{
			m_doStop = false;
			m_mutex.unlock();
			break;
		}
		m_mutex.unlock();

		if(!m_cap.grab())
			continue;

		// Retrieve frames
		m_cap.retrieve(m_grabbedFrame);
		cv::cvtColor(m_grabbedFrame,
		  		     m_currentFrame,
				     CV_BGR2GRAY);

		// Convert grabbed frames to QImage
		m_frame = TW::Mat2QImage(m_currentFrame);
		emit newFrame(m_frame);
	}
}