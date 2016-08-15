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
	///\brief Constructor for the non-CPU_ICGN computation mode
	OneCamWidget(
		int deviceNumber,
		ImageBufferPtr refImgBuffer,
		ImageBufferPtr tarImgBuffer,
		int iImgWidth,
		int iImgHeight,
		int iNumberX,
		int iNumberY,
		const QRect& roi,
		// Add computation Mode
		ComputationMode computationMode = ComputationMode::GPUFFTCC,
		QWidget *parent = 0);

	///\brief Constructor for the CPU-ICGN computation mode, because the ICGN
	/// needs another set of ref & targ image buffers for its own use.
	OneCamWidget(
		int deviceNumber,
		ImageBufferPtr refImgBuffer,
		ImageBufferPtr tarImgBuffer,
		ImageBufferPtr refImgBufferCPU_ICGN,
		ImageBufferPtr tarImgBufferCPU_ICGN,
		const int iNumICGNThreads,
		int iImgWidth,
		int iImgHeight,
		int iNumberX,
		int iNumberY,
		const QRect& roi,
		ComputationMode computationMode = ComputationMode::GPUFFTCC,
		QWidget *parent = 0);

	~OneCamWidget();

	bool connectToCamera(
		bool ifDropFrame,
		int iSubsetX, int iSubsetY,
		int iGridSpaceX, int iGridSpaceY,
		int iMarginX, int iMarginY,
		const QRect& roi);


private:
	void stopCaptureThread();
	void stopFFTCCWorkerThread();
	void stopICGNWorkerThread();

signals:
	void titleReady(const QString&);

	public slots:
	void updateRefFrame(const QImage&);	// signal: &capturethread::newRefFrame
	void updateTarFrame(const QImage&);	// signal: &capturethread::newTarFrame
	void updateStatics(const int&, const int&);
	void testSlot(const float&);

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
	
	// These three parameters will only be used if CPU ICGN is the computation mode
	ImageBufferPtr m_refImgBufferCPU_ICGN;	// Ref Image buffer for the CPU ICGN
	ImageBufferPtr m_tarImgBufferCPU_ICGN;  // Tar Image buffer for the CPU ICGN
	VecBufferfPtr m_fUBuffer;
	VecBufferfPtr m_fVBuffer;
	VecBufferiPtr m_iPOIXYBuffer;
	int m_iNumICGNThreads;

	int m_iImgWidth;
	int m_iImgHeight;
	int m_iNumberX;
	int m_iNumberY;

	ComputationMode m_computationMode;
};

#endif // ONECAMWIDGET_H
