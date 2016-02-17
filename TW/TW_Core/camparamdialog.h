#ifndef CAMPARAMDIALOG_H
#define CAMPARAMDIALOG_H

#include <QDialog>
#include <QRect>
#include "ui_camparamdialog.h"

#include "Structures.h"
#include "camparamthread.h"

class CamParamDialog : public QDialog
{
	Q_OBJECT

public:
	CamParamDialog(QWidget *parent = 0);
	~CamParamDialog();

	bool connectToCamera(int width, int height);
	QRect GetROI() const { return m_ROIRect; }

private:
	void stopCaptureThread();

public slots:
	void newMouseData(const MouseData& mouseData);
	void updateMouseCursorPosLabel();
	void updateFrame(const QImage &frame);

signals:
	void setROI(QRect roi);

private:
	QRect m_ROIRect;
	QScopedPointer<CamParamThread> m_captureThread; 
	bool m_isCameraConnected;

	Ui::CamParamDialog ui;
};

#endif // CAMPARAMDIALOG_H
