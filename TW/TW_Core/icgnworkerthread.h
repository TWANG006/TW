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
	// Disable the default constructors
	ICGNWorkerThread() = delete;
	ICGNWorkerThread(const ICGNWorkerThread&) = delete;
	ICGNWorkerThread& operator=(const ICGNWorkerThread&) = delete;

	ICGNWorkerThread(
		ImageBufferPtr refImgBuffer,
		ImageBufferPtr tarImgBuffer,
		VecBufferfPtr fUBuffer,
		VecBufferfPtr fVBuffer,
		VecBufferiPtr iPOIXYBuffer,
		const QRect &roi,
		int iWidth, int iHeight,
		int iNumberX, int iNumberY,
		int iSubsetX, int iSubsetY,
		int iNumbIterations,
		float fDeltaP);
	~ICGNWorkerThread();

public slots:
	void processFrame();

signals:
	void testSignal(const float&);

private:
	ICGN2DPtr m_ICGN2DPtr;
	ImageBufferPtr m_refImgBuffer;
	ImageBufferPtr m_tarImgBuffer;
	VecBufferfPtr m_fUBuffer;
	VecBufferfPtr m_fVBuffer;
	VecBufferiPtr m_iPOIXYBuffer;
	std::vector<int> m_iNumberIterations;
};

#endif // ICGNWORKERTHREAD_H
