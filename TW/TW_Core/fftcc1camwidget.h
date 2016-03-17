#ifndef FFTCC1CAMWIDGET_H
#define FFTCC1CAMWIDGET_H

#include <QWidget>
#include <QThread>
#include <QPointer>
#include "ui_fftcc1camwidget.h"

#include "Structures.h"
#include "capturethread.h"
#include "fftcctworkerthread.h"


class FFTCC1CamWidget : public QWidget
{
	Q_OBJECT

public:
	FFTCC1CamWidget(int deviceNumber,
					ImageBufferPtr refImgBuffer,
					ImageBufferPtr tarImgBuffer,
					QWidget *parent = 0);
	~FFTCC1CamWidget();

	bool connectToCamera(bool ifDropFrame, 
						 int width, int height,
						 int iSubsetX, int iSubsetY,
						 int iGridSpaceX, int iGridSpaceY,
						 int iMarginX, int iMarginY,
						 const QRect& roi);


private:
	void stopCaptureThread();
	void stopFFTCCWorkerThread();

public slots:
	void updateRefFrame(const QImage&);				// signal: &capturethread::newRefFrame
	void updateTarFrame(const QImage&);				// signal: &capturethread::newTarFrame

private:
	Ui::FFTCC1CamWidget ui;

	QThread m_fftccWorkerThread;					// FFTCC thread
	QPointer<CaptureThread> m_captureThread;		// Capture thread	
	QPointer<FFTCCTWorkerThread	> m_fftccWorker;	// FFTCC worker

	bool m_isCameraConnected;
	int m_iDeviceNumber;
	ImageBufferPtr m_refImgBuffer;
	ImageBufferPtr m_tarImgBuffer;
};

#endif // FFTCC1CAMWIDGET_H
