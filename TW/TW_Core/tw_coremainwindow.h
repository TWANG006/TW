#ifndef TW_COREMAINWINDOW_H
#define TW_COREMAINWINDOW_H

#include <QtWidgets/QMainWindow>
#include <QPointer>
#include "ui_tw_coremainwindow.h"

#include "camparamdialog.h"
#include "fftcc1camwidget.h"

class TW_CoreMainWindow : public QMainWindow
{
	Q_OBJECT

public:
	TW_CoreMainWindow(QWidget *parent = 0);
	~TW_CoreMainWindow();
	
protected:
	void closeEvent(QCloseEvent *event) Q_DECL_OVERRIDE;

protected slots:
	void OnOpenImgFile();
	void OnCapture_From_Camera();

private:
	Ui::TW_CoreMainWindowClass ui;

	QString qstrLastSelectedDir;						//!- Hold last opend directory
	QPointer<CamParamDialog>   m_camParamDialog;
	QPointer<FFTCC1CamWidget>  m_fftcc1camWidget;
	QRect m_ROI;
	bool m_isDropFrameChecked;
	int m_iSubsetX;
	int m_iSubsetY;
	int m_iMarginX;
	int m_iMarginY;
	int m_iGridSpaceX;
	int m_iGridSpaceY;

};

#endif // TW_COREMAINWINDOW_H
