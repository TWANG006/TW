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
	int GetSubetX() const { return ui.SubsetX_lineEdit->text().toInt();}
	int GetSubetY() const { return ui.SubsetY_lineEdit->text().toInt();}
	int GetMarginX() const { return ui.MarginX_lineEdit->text().toInt();}
	int GetMarginY() const { return ui.MarginY_lineEdit->text().toInt();}
	int GetGridX() const { return ui.GridX_lineEdit->text().toInt();}
	int GetGridY() const { return ui.GridY_lineEdit->text().toInt();}
	int GetRefImgBufferSize() const { return ui.refImgBufferlineEdit->text().toInt(); }
	int GetTarImgBufferSize() const{ return ui.tarImgBufferlineEdit->text().toInt(); }
	int GetInputSourceWidth();
	int GetInputSourceHeight();
	bool isDropFrame() const { return ui.dropFrame_checkBox->isChecked(); }
	ComputationMode GetComputationMode();
	int GetNumICGNThreads() const { return ui.ICGNspinBox->value();}

	int ComputeNumberofPOIs();
	int ComputeNumberofPOIsX();
	int ComputeNumberofPOIsY();

private:
	void stopCaptureThread();

public slots:
	void newMouseData(const MouseData& mouseData);
	void updateMouseCursorPosLabel();
	void updatePOINums(const QString&);
	void updatePOINumsHelper();

private slots:
	void updateFrame(const QImage &frame);
	void updateThreadStats(const ThreadStatisticsData &statData);

signals:
	void setROI(QRect roi);

private:
	QRect m_ROIRect;
	CamParamThread *m_captureThread; 
	bool m_isCameraConnected;

	Ui::CamParamDialog ui;
};

#endif // CAMPARAMDIALOG_H
