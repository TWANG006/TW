#ifndef FFTCCTWORKERTHREAD_H
#define FFTCCTWORKERTHREAD_H

#include <QObject>
#include "Structures.h"

#include "TW_paDIC_cuFFTCC2D.h"

class FFTCCTWorkerThread : public QObject
{
	Q_OBJECT

public:
	FFTCCTWorkerThread(const QRect &roi, QObject *parent);
	~FFTCCTWorkerThread();

public slots:
	void processFrame(int iFrameCount);

private:

	
};

#endif // FFTCCTWORKERTHREAD_H
