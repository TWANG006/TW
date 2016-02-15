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

public slots:
	void newMouseData(const MouseData& mouseData);
	void updateMouseCursorPosLabel();
	void updateFrame(const QImage &frame);

private:
	QRect m_ROIRect;
	QScopedPointer<CamParamThread> m_captureThread; 

	Ui::CamParamDialog ui;
};

#endif // CAMPARAMDIALOG_H
