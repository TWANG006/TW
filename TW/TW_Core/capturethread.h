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
	using ImageBuffer = TW::Concurrent_Buffer<cv::Mat>;
	using ImageBufferPtr = std::shared_ptr<ImageBuffer>;

public:
	CaptureThread(ImageBufferPtr imgBuffer,
				  bool isDropFrameIfBufferFull,
  				  int iDeviceNumber,
				  int width,
				  int height,
				  QObject *parent);
	~CaptureThread();

	bool connectToCamera();
	bool disconnectCamera();
	void stop();

protected:
	void run() Q_DECL_OVERRIDE;

private:
	cv::VideoCapture m_cap;
	cv::Mat m_grabbedFrame;
	ImageBufferPtr m_imgBuffer;
	volatile bool m_isAboutToStop;
	QMutex m_stopMutex;
	int m_iDeviceNumber;
	int m_iWidth;
	int m_iHeight;
	bool m_isDropFrameIfBufferFull;

signals:
	void updateRefImg();
};

#endif // CAPTURETHREAD_H
