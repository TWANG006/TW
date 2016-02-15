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

protected:
	void run() Q_DECL_OVERRIDE;

public slots:
	void setROI(QRect roi);

signals:
	void newFrame(const QImage& frame);


private:
	QMutex m_mutex;
	volatile bool m_doStop;
	QTime m_time;

	cv::VideoCapture m_cap;
	cv::Mat m_grabbedFrame;
	cv::Mat m_currentFrame;
	QImage m_frame;
	
	int m_width;
	int m_height;

	/*std::shared_ptr<TW::SharedImageBuffer> m_sharedImgBuffer;*/
};

#endif // CAMPARAMTHREAD_H
