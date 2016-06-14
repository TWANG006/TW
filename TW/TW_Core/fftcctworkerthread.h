#ifndef FFTCCTWORKERTHREAD_H
#define FFTCCTWORKERTHREAD_H

#include <QObject>
#include "Structures.h"

#include "TW.h"
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
	TW::real_t *m_d_fU;
	TW::real_t *m_d_fV;
	TW::real_t *m_d_fAccumulateU;
	TW::real_t *m_d_fAccumulateV;

	unsigned int *m_d_UColorMap;
	unsigned int *m_d_VColorMap;

	TW::real_t *m_d_fMaxU;
	TW::real_t *m_d_fMinU;
	TW::real_t *m_d_fMaxV;
	TW::real_t *m_d_fMinV;

	int *m_d_iCurrentPOIXY;
	TW::real_t *m_d_fZNCC;

	int m_iNumberX;
	int m_iNumberY;
	int m_iNumPOIs;
	int m_iWidth;
	int m_iHeight;
	int m_iSubsetX;
	int m_iSubsetY;


	cuFftcc2DPtr m_Fftcc2DPtr;
	ImageBufferPtr m_refImgBuffer;
	ImageBufferPtr m_tarImgBuffer;
	QRect m_ROI;
	std::shared_ptr<SharedResources> m_sharedResources;
};

#endif // FFTCCTWORKERTHREAD_H
