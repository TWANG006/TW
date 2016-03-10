#include "fftcctworkerthread.h"

FFTCCTWorkerThread::FFTCCTWorkerThread(const QRect &roi,
									   QObject *parent)
	: QObject(parent)
{}

FFTCCTWorkerThread::~FFTCCTWorkerThread()
{

}

void FFTCCTWorkerThread::processFrame(int iFrameCount)
{
	if(iFrameCount % 50 ==1)
		;


}