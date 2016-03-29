#include "camparamthread.h"

#include "TW_MatToQImage.h"

CamParamThread::CamParamThread(int width, 
							   int height)
	: m_width(width)
	, m_height(height)
	, m_doStop(false)
	, m_sampleNumber(0)
	, m_fpsSum(0)
{
	m_fpsQueue.clear();
	m_statsData.averageFPS = 0;
	m_statsData.nFramesProcessed = 0;
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
	QMutexLocker locker(&m_otherMutex);
	m_currentROI.x = roi.x();
	m_currentROI.y = roi.y();
	m_currentROI.width = roi.width();
	m_currentROI.height = roi.height();
}

QRect CamParamThread::GetCurrentROI()
{
	return QRect(m_currentROI.x, 
				 m_currentROI.y,
				 m_currentROI.width,
				 m_currentROI.height);
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

		// Save the capture time
		m_captureTime = m_time.elapsed();
		m_time.start();

		if(!m_cap.grab())
			continue;

		// Retrieve frames
		if(m_cap.retrieve(m_grabbedFrame))
			m_currentFrame = cv::Mat(m_grabbedFrame.clone(), m_currentROI);
				
		/*cv::cvtColor(m_currentFrame,
		  		     m_currentGrayFrame,
				     CV_BGR2GRAY);*/

		// Convert grabbed frames to QImage
		m_frame = TW::Mat2QImage(m_currentFrame);
		emit newFrame(m_frame);
		
		// Update statistics
		updateFPS(m_captureTime);
		m_statsData.nFramesProcessed++;
		
		emit updateStatisticsInGUI(m_statsData);
	}
	qDebug() << "[Param Setting] Stopping capture thread...";
}

void CamParamThread::updateFPS(int timeElapsed)
{
	// Compute the average FPS per 32 frames
	if(timeElapsed > 0)
	{
		m_fpsQueue.enqueue((int)1000/timeElapsed);
		m_sampleNumber++;
	}

	if(m_fpsQueue.size() > 32)
	{
		m_fpsQueue.dequeue();
	}

	if(m_fpsQueue.size() == 32 && m_sampleNumber == 32)
	{
		while(!m_fpsQueue.empty())
			m_fpsSum += m_fpsQueue.dequeue();

		// Calculate average FPS
		m_statsData.averageFPS = m_fpsSum / 32;
		m_fpsSum = 0;
		m_sampleNumber = 0;
	}
}