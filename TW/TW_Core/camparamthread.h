#ifndef CAMPARAMTHREAD_H
#define CAMPARAMTHREAD_H

#include <QThread>
#include <QImage>
#include <QTime>

#include <memory>
#include <opencv2\opencv.hpp>
#include "TW.h"
#include "SharedImageBuffer.h"

#include "Structures.h"

/// \brief Thread class for capturing images from camera.
/// \mtehods:
/// setROI: set the current ROI of the captured image
/// 

class CamParamThread : public QThread
{
	Q_OBJECT

public:
	CamParamThread(int width, 
				   int height);
	~CamParamThread();

	bool connectToCamera();
	bool disconnectCamera();
	bool isCameraConnected();
	int getInputSourceWidth();
    int getInputSourceHeight();
	void stop();
	QRect GetCurrentROI();

protected:
	void run() Q_DECL_OVERRIDE;

public slots:
	void setROI(QRect roi);

signals:
	void newFrame(const QImage& frame);
	void updateStatisticsInGUI(const ThreadStatisticsData &statData);

private:
	void updateFPS(int);
	QMutex m_mutex;
	QMutex m_otherMutex;
	volatile bool m_doStop;
	QTime m_time;

	cv::VideoCapture m_cap;
	cv::Mat m_grabbedFrame;
	cv::Mat m_currentFrame;
	/*cv::Mat m_currentGrayFrame;*/
	QImage m_frame;

	ThreadStatisticsData m_statsData;
	QQueue<int> m_fpsQueue;
	int m_sampleNumber;
    int m_fpsSum;
	
	cv::Rect m_currentROI;
	int m_width;
	int m_height;
	int m_captureTime;

	/*std::shared_ptr<TW::SharedImageBuffer> m_sharedImgBuffer;*/
};

#endif // CAMPARAMTHREAD_H
