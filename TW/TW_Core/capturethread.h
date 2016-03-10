#ifndef CAPTURETHREAD_H
#define CAPTURETHREAD_H

#include <memory>
#include <QThread>
#include <opencv2\opencv.hpp>

#include "Structures.h"
#include "TW_Concurrent_Buffer.h"

class CaptureThread : public QThread
{
	Q_OBJECT

public:
	CaptureThread(ImageBufferPtr refImgBuffer,
				  ImageBufferPtr tarImgBuffer,
				  bool isDropFrameIfBufferFull,
  				  int iDeviceNumber,
				  int width,
				  int height,
				  QObject *parent);
	~CaptureThread();

	bool connectToCamera();
	bool disconnectCamera();
	bool isCameraConnected()    const { return m_cap.isOpened(); }
	int  getInputSourceWidth()  const { return m_cap.get(CV_CAP_PROP_FRAME_WIDTH); }
	int  getInputSourceHeight() const { return m_cap.get(CV_CAP_PROP_FRAME_HEIGHT); }

	void stop();

protected:
	void run() Q_DECL_OVERRIDE;

private:
	cv::VideoCapture m_cap;
	cv::Mat m_grabbedFrame;
	cv::Mat m_currentFrame;
	ImageBufferPtr m_tarImgBuffer;
	ImageBufferPtr m_refImgBuffer;
	volatile bool m_isAboutToStop;
	QMutex m_stopMutex;
	int m_iDeviceNumber;
	int m_iWidth;
	int m_iHeight;
	int m_iFrameCount;
	bool m_isDropFrameIfBufferFull;

signals:
	void newTarFrame(int frameCount);
};

#endif // CAPTURETHREAD_H
