#include "tw_coremainwindow.h"
#include <QFileDialog>

TW_CoreMainWindow::TW_CoreMainWindow(QWidget *parent)
	: QMainWindow(parent), qstrLastSelectedDir(tr("/home"))
{
	ui.setupUi(this);

	connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(OpenImgFile()));
}

TW_CoreMainWindow::~TW_CoreMainWindow()
{

}

void TW_CoreMainWindow::OpenImgFile()
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