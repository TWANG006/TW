#ifndef ONECAMWIDGET_H
#define ONECAMWIDGET_H

#include <QWidget>
#include <QThread>
#include <QPointer>
#include "ui_onecamwidget.h"

#include "Structures.h"
#include "capturethread.h"
#include "fftcctworkerthread.h"
#include "icgnworkerthread.h"
#include "glwidget.h"

class OneCamWidget : public QWidget
{
	Q_OBJECT

public:
	OneCamWidget(int deviceNumber,
					ImageBufferPtr refImgBuffer,
					ImageBufferPtr tarImgBuffer,
					int iImgWidth,
					int iImgHeight,
					const QRect& roi,
					// Add computation Mode
					ComputationMode computationMode = ComputationMode::GPUFFTCC,
					QWidget *parent = 0);
	~OneCamWidget();

	bool connectToCamera(bool ifDropFrame, 
						 int iSubsetX, int iSubsetY,
						 int iGridSpaceX, int iGridSpaceY,
						 int iMarginX, int iMarginY,
						 const QRect& roi);


private:
	void stopCaptureThread();
	void stopFFTCCWorkerThread();

signals:
	void titleReady(const QString&);

public slots:
	void updateRefFrame(const QImage&);	// signal: &capturethread::newRefFrame
	void updateTarFrame(const QImage&);	// signal: &capturethread::newTarFrame
	void updateStatics(const int&, const int&);

private:
	Ui::OneCamWidget ui;
	GLWidget *m_twGLwidget;

	CaptureThread *m_captureThread;		// Capture thread	
	QThread *m_fftccWorkerThread;	    // FFTCC thread
	FFTCCTWorkerThread *m_fftccWorker;	// FFTCC worker
	QThread *m_icgnWorkerThread;		// ICGN thread
	ICGNWorkerThread *m_icgnWorker;		// ICGN worker
	

	std::shared_ptr<SharedResources> m_sharedResources;

	bool m_isCameraConnected;
	int m_iDeviceNumber;
	ImageBufferPtr m_refImgBuffer;
	ImageBufferPtr m_tarImgBuffer;
	int m_iImgWidth;
	int m_iImgHeight;

	ComputationMode m_computationMode;
};

#endif // ONECAMWIDGET_H
