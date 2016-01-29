#ifndef CAMPARAMTHREAD_H
#define CAMPARAMTHREAD_H

#include <QThread>

class CamParamThread : public QThread
{
	Q_OBJECT

public:
	CamParamThread(QObject *parent);
	~CamParamThread();

private:
	
};

#endif // CAMPARAMTHREAD_H
