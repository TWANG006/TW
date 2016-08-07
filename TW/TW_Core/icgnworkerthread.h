#ifndef ICGNWORKERTHREAD_H
#define ICGNWORKERTHREAD_H

/* Thread to perform the ICGN in the background threads. 
   This thread is enabled when the ComputationMode::GPUFFTCC_CPUICGN
   is selected.
*/

#include <QObject>
#include "Structures.h"
#include "TW_paDIC_ICGN2D_CPU.h"


class ICGNWorkerThread : public QObject
{
	Q_OBJECT

public:
	ICGNWorkerThread(
		ImageBufferPtr refImgBuffer,
		ImageBufferPtr tarImgBuffer,
		int iWidth, int iHeight,
		int iSubsetX, int iSubsetY,
		int iGridSpaceX, int iGridSpaceY,
		int iMarginX, int iMarginY,
		const QRect &roi);
	~ICGNWorkerThread();

public slots:
	void processFrame();

signals:

private:
	ICGN2DPtr m_ICGN2DPtr;
	ImageBufferPtr m_refImgBuffer;
	ImageBufferPtr m_tarImgBuffer;
};

#endif // ICGNWORKERTHREAD_H
