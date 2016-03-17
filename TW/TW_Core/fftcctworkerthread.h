#ifndef FFTCCTWORKERTHREAD_H
#define FFTCCTWORKERTHREAD_H

#include <QObject>
#include "Structures.h"

#include "TW_paDIC_cuFFTCC2D.h"

class FFTCCTWorkerThread : public QObject
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
					   const cv::Mat &firstFrame);
	~FFTCCTWorkerThread();

public slots:
	void processFrame(const int &iFrameCount);


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
};

#endif // FFTCCTWORKERTHREAD_H
