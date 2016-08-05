#ifndef ICGNWORKERTHREAD_H
#define ICGNWORKERTHREAD_H

#include <QObject>
#include "Structures.h"


class ICGNWorkerThread : public QObject
{
	Q_OBJECT

public:
	ICGNWorkerThread(QObject *parent);
	~ICGNWorkerThread();

public slots:
	void processFrame();

signals:

private:
	ICGN2DPtr m_ICGN2DPtr;	
};

#endif // ICGNWORKERTHREAD_H
