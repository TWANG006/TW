#include "tw_coremainwindow.h"
#include <QFileDialog>
#include <QDebug>

TW_CoreMainWindow::TW_CoreMainWindow(QWidget *parent)
	: QMainWindow(parent)
	, qstrLastSelectedDir(tr("/home"))
	, m_camParamDialog(nullptr)
{
	ui.setupUi(this);

	connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(OnOpenImgFile()));
	connect(ui.actionCapture_From_Camra, SIGNAL(triggered()), this, SLOT(OnCapture_From_Camera()));

}

TW_CoreMainWindow::~TW_CoreMainWindow()
{

}

void TW_CoreMainWindow::OnOpenImgFile()
{
	QStringList imgFileList = QFileDialog::getOpenFileNames(
		this,
		tr("Select one or more files to open"),
		qstrLastSelectedDir,
		tr("Images (*.png *.xpm *.jpg *.bmp *.tif)"));

	//!- Make sure image(s) is selected
	if (!imgFileList.isEmpty())
	{
		//!- save image's directory
		qstrLastSelectedDir = imgFileList.at(0);

		//!- Load images into QImage objects

	}
}

void TW_CoreMainWindow::OnCapture_From_Camera()
{
	m_camParamDialog.reset(new CamParamDialog(this));

	m_camParamDialog->connectToCamera(-1,-1);

	if(m_camParamDialog->exec() == QDialog::Accepted)
	{
		qDebug()<<"<<<";

	}
	
	m_camParamDialog.reset(nullptr);
}