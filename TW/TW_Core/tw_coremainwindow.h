#ifndef TW_COREMAINWINDOW_H
#define TW_COREMAINWINDOW_H

#include <QtWidgets/QMainWindow>
#include <QPointer>
#include "ui_tw_coremainwindow.h"

#include "TW_Concurrent_Buffer.h"
#include "camparamdialog.h"
#include "onecamwidget.h"
#include "capturethread.h"

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
	void updateTitle(const QString&);
	//void OnFrames(int num);

private:
	Ui::TW_CoreMainWindowClass ui;

	ImageBufferPtr m_refBuffer;						// Global refImg buffer
	ImageBufferPtr m_tarBuffer;						// Global tarImg 
	ImageBufferPtr m_refBufferCPU_ICGN;
	ImageBufferPtr m_tarBufferCPU_ICGN;			

	QString qstrLastSelectedDir;					//!- Hold last opend directory
	CamParamDialog  *m_camParamDialog;
	OneCamWidget *m_onecamWidget;
	//CaptureThread *m_testCap;
	QRect m_ROI;
	bool m_isDropFrameChecked;
	int m_iSubsetX;
	int m_iSubsetY;
	int m_iMarginX;
	int m_iMarginY;
	int m_iGridSpaceX;
	int m_iGridSpaceY;
	int m_iImgWidth;
	int m_iImgHeight;

	// Global parameters for the computation of FFTCC & ICGN
	int m_d_iU;
	int m_d_iV;
	float m_d_fZNCC;

	ComputationMode m_computationMode;
};

#endif // TW_COREMAINWINDOW_H
