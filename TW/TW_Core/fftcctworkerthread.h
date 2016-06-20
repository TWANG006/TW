#ifndef FFTCCTWORKERTHREAD_H
#define FFTCCTWORKERTHREAD_H

#include <QObject>
#include <QTime>
#include <QQueue>
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
	

signals:
	void frameReady();
	void runningStaticsReady(const int& iNumPOIs,
							 const int& iFPS);

private:
	 void updateFPS(int);

private:
	TW::real_t *m_d_fU;				// Newly computed U
	TW::real_t *m_d_fV;				// Newly computed 
	TW::real_t *m_d_fZNCC;			// Newly computed ZNCC coefficients

	TW::real_t *m_d_fAccumulateU;	// Accumulated U w.r.t the first reference image
	TW::real_t *m_d_fAccumulateV;	// Accumulated V w.r.t the first reference image

	unsigned int *m_d_UColorMap;	// Color map of the U component
	unsigned int *m_d_VColorMap;	// Color map of the V component

	/*
		Max & min values of the U and V
		These values are only needed when there's a need for dynamic range of
		U and V. Currently the range is fixed as [-20, 20] pixels
	*/
	TW::real_t *m_d_fMaxU;		
	TW::real_t *m_d_fMinU;
	TW::real_t *m_d_fMaxV;
	TW::real_t *m_d_fMinV;

	int *m_d_iCurrentPOIXY;	// Updated POI positions in the target image
	
	/*
		Launch paramters
	*/
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

	/*Running statics*/
	QTime m_t;
	QQueue<int> m_fps;
	int m_averageFPS;
	int m_processingTime;
	int m_fpsSum;
	int m_sampleNumber;

};

#endif // FFTCCTWORKERTHREAD_H
