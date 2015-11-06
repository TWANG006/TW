#ifndef TW_COREMAINWINDOW_H
#define TW_COREMAINWINDOW_H

#include <QtWidgets/QMainWindow>
#include "ui_tw_coremainwindow.h"

class TW_CoreMainWindow : public QMainWindow
{
	Q_OBJECT

public:
	TW_CoreMainWindow(QWidget *parent = 0);
	~TW_CoreMainWindow();
	
protected slots:
	void OpenImgFile();

private:
	Ui::TW_CoreMainWindowClass ui;
	QString qstrLastSelectedDir;			//!- Hold last opend directory
	QMdiSubWindow *m_ActiveSubWindow;		//!- Active image window
	
};

#endif // TW_COREMAINWINDOW_H
