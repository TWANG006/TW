#include "icgnworkerthread.h"

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

}