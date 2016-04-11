#ifndef FFTCCTWORKERTHREAD_H
#define FFTCCTWORKERTHREAD_H

#include <QObject>
#include "Structures.h"

#include "TW_paDIC_cuFFTCC2D.h"

class FFTCCTWorkerThread : public QObject, protected QOpenGLFunctions_3_3_Core
{
	Q_OBJECT

public:
	FFTCCTWorkerThread(// Inputs
					   ImageBufferPtr refImgBuffer,
					   ImageBufferPtr tarImgBuffer,
					   int iWidth, int iHeight,
					   int iSubsetX, int iSubsetY,
					   int iGridSpaceX, int iGridSpaceY,
					   int iMarginX, int iMarginY,
					   const QRect &roi,
					   const cv::Mat &firstFrame,
					   std::shared_ptr<SharedResources>&);
	~FFTCCTWorkerThread();

public slots:
	void processFrame(const int &iFrameCount);
	void render();

signals:
	void frameReady();

private:
	int *m_d_iU;
	int *m_d_iV;
	float *m_d_fZNCC;
	int m_iWidth;
	int m_iHeight;
	cuFftcc2DPtr m_Fftcc2DPtr;
	ImageBufferPtr m_refImgBuffer;
	ImageBufferPtr m_tarImgBuffer;
	QRect m_ROI;
	std::shared_ptr<SharedResources> m_sharedResources;
};

#endif // FFTCCTWORKERTHREAD_H
