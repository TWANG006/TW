#include "icgnworkerthread.h"
#include <omp.h>

ICGNWorkerThread::ICGNWorkerThread(
	ImageBufferPtr refImgBuffer,
	ImageBufferPtr tarImgBuffer,
	int iWidth, int iHeight,
	int iSubsetX, int iSubsetY,
	int iGridSpaceX, int iGridSpaceY,
	int iMarginX, int iMarginY,
	const QRect &roi)
	: m_ICGN2DPtr(nullptr)
{

}

ICGNWorkerThread::~ICGNWorkerThread()
{

}

void ICGNWorkerThread::processFrame()
{
	float j = 0;
#pragma omp parallel for
	for(int i=0; i<1000000; i++)
	{
		j += 0.5;
	}
	emit testSignal(j);
}