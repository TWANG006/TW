#ifndef CAPTURETHREAD_H
#define CAPTURETHREAD_H

#include <memory>
#include <QThread>
#include <QImage>
#include <opencv2\opencv.hpp>

#include "Structures.h"
#include "TW_Concurrent_Buffer.h"

class CaptureThread : public QThread
{
	Q_OBJECT

public:
	CaptureThread() = delete;
	CaptureThread(const CaptureThread&) = delete;
	CaptureThread& operator=(const CaptureThread&) = delete;

	CaptureThread(
		ImageBufferPtr refImgBuffer,
		ImageBufferPtr tarImgBuffer,
		bool isDropFrameIfBufferFull,
		int iDeviceNumber,
		int width,
		int height,
		ComputationMode computationMode,
		QObject *parent);

	CaptureThread(
		ImageBufferPtr refImgBuffer,
		ImageBufferPtr tarImgBuffer,
		ImageBufferPtr refImgBufferCPU_ICGN,
		ImageBufferPtr tarImgBufferCPU_ICGN,
		bool isDropFrameIfBufferFull,
		int iDeviceNumber,
		int width,
		int height,
		ComputationMode computationMode,
		QObject *parent);

	~CaptureThread();

	/// \brief Grab a first frame for the initialziation of the cuFFTCC2D algorithm
	bool grabTheFirstRefFrame(cv::Mat &firstFrame);

	/// \brife determine whether the camera is connected or not. This function should
	/// be called before any other methods within the CaptureThread except the 
	/// grabTheFirstRefFrame one
	bool connectToCamera();

	/// \brief Call this function to stop the thread and close the camera connection
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
	cv::Mat m_grayFrame;
	ImageBufferPtr m_tarImgBuffer;
	ImageBufferPtr m_refImgBuffer;
	ImageBufferPtr m_tarImgBufferCPU_ICGN;
	ImageBufferPtr m_refImgBufferCPU_ICGN;
	volatile bool m_isAboutToStop;
	QMutex m_stopMutex;
	int m_iDeviceNumber;
	int m_iWidth;
	int m_iHeight;
	int m_iFrameCount;
	bool m_isDropFrameIfBufferFull;
	ComputationMode m_computationMode;

	// For the GUI use to update the ref & tar images
	QImage m_Qimg;

signals:
	void newTarFrame(const int &frameCount);
	void newRefQImg(const QImage& refImg);
	void newTarQImg(const QImage& tarImg);
};

#endif // CAPTURETHREAD_H
