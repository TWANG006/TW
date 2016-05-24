#ifndef ICCGNWORKERTHREAD_H
#define ICCGNWORKERTHREAD_H

#include <QObject>
#include "Structures.h"


class ICCGNWorkerThread : public QObject
{
	Q_OBJECT

public:
	ICCGNWorkerThread(QObject *parent);
	~ICCGNWorkerThread();

public slots:
	void processFrame();

signals:

private:
	ICGN2DPtr m_ICGN2DPtr;	
};

#endif // ICCGNWORKERTHREAD_H
