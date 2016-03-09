#ifndef FFTCCTWORKERTHREAD_H
#define FFTCCTWORKERTHREAD_H

#include <QObject>

class FFTCCTWorkerThread : public QObject
{
	Q_OBJECT

public:
	FFTCCTWorkerThread(QObject *parent);
	~FFTCCTWorkerThread();

private:
	
};

#endif // FFTCCTWORKERTHREAD_H
