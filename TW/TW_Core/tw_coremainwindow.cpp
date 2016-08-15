#include "tw_coremainwindow.h"
#include <QFileDialog>
#include <QDebug>
#include <QMessageBox>

TW_CoreMainWindow::TW_CoreMainWindow(QWidget *parent)
	: QMainWindow(parent)
	, qstrLastSelectedDir(tr("/home"))
	, m_camParamDialog(nullptr)
	, m_onecamWidget(nullptr)
	, m_isDropFrameChecked(true)
	, m_iSubsetX(0), m_iSubsetY(0)
	, m_iMarginX(0), m_iMarginY(0)
	, m_iGridSpaceX(0), m_iGridSpaceY(0)
	, m_iImgWidth(0), m_iImgHeight(0)
	, m_refBuffer(nullptr)
	, m_tarBuffer(nullptr)
	, m_refBufferCPU_ICGN(nullptr)
	, m_tarBufferCPU_ICGN(nullptr)
{
	ui.setupUi(this);
	/*m_testCap = new CaptureThread(m_refBuffer,m_testCap, false,0,-1,-1,this);*/

	connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(OnOpenImgFile()));
	connect(ui.actionCapture_From_Camra, &QAction::triggered, this, &TW_CoreMainWindow::OnCapture_From_Camera);
	/*connect(m_testCap, &CaptureThread::newTarFrame, this, &TW_CoreMainWindow::OnFrames);

	m_testCap->connectToCamera();

	m_testCap->start();*/
}

//void TW_CoreMainWindow::OnFrames(int num)
//{
//	if(num%50==1)
//	{
//		qDebug()<<num<<"Cao";
//		m_refBuffer->DeQueue();
//	}
//	qDebug()<<num;
//	m_tarBuffer->DeQueue();
//}	

TW_CoreMainWindow::~TW_CoreMainWindow()
{
	/*m_testCap->stop();
	m_testCap->wait();*/
}


//void TW_CoreMainWindow::closeEvent(QCloseEvent *event)
//{
//	ui.mdiArea->closeAllSubWindows();
//	   if (ui.mdiArea->currentSubWindow()) {
//        event->ignore();
//    } else {
//        event->accept();
//    }
//}

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
	// Open the Camera Parameters dialog to set parameters
	// needed for the next computations
	QLayoutItem *child;
	while ((child = ui.gridLayout->takeAt(0)) != 0) {
		delete child->widget();
		delete child;
		if (m_camParamDialog != nullptr)
		{
			delete m_camParamDialog;
			m_camParamDialog = nullptr;
		}

	}

	m_camParamDialog = new CamParamDialog(this);

	// Set the width & height to -1 to accept the default resolution of the 
	// camera or set the resolution by hand.
	if (!m_camParamDialog->connectToCamera(640, 480))
	{
		QMessageBox::critical(this,
			tr("Fail!"),
			tr("[Param Setting] Cannot connect to camera!"));
		return;
	}

	//!- If OK button is clicked, accept all the settings
	if (m_camParamDialog->exec() == QDialog::Accepted)
	{
		// Get parameters from the camera parameter setting dialog
		m_ROI = m_camParamDialog->GetROI();
		m_iSubsetX = m_camParamDialog->GetSubetX();
		m_iSubsetY = m_camParamDialog->GetSubetY();
		m_iMarginX = m_camParamDialog->GetMarginX();
		m_iMarginY = m_camParamDialog->GetMarginY();
		m_iGridSpaceX = m_camParamDialog->GetGridX();
		m_iGridSpaceY = m_camParamDialog->GetGridY();
		m_isDropFrameChecked = m_camParamDialog->isDropFrame();
		m_iImgWidth = m_camParamDialog->GetInputSourceWidth();
		m_iImgHeight = m_camParamDialog->GetInputSourceHeight();
		m_computationMode = m_camParamDialog->GetComputationMode();
		int iNumberX = m_camParamDialog->ComputeNumberofPOIsX();
		int iNumberY = m_camParamDialog->ComputeNumberofPOIsY();
		int iNumICGNThreads = m_camParamDialog->GetNumICGNThreads();
		m_refBuffer.reset(new TW::Concurrent_Buffer<cv::Mat>(m_camParamDialog->GetRefImgBufferSize()));
		m_tarBuffer.reset(new TW::Concurrent_Buffer<cv::Mat>(m_camParamDialog->GetTarImgBufferSize()));

		// Enable the ICGN image buffers if CPU-ICGN computation mode is on
		if (m_computationMode == ComputationMode::GPUFFTCC_CPUICGN)
		{
			m_refBufferCPU_ICGN.reset(new TW::Concurrent_Buffer<cv::Mat>(20));
			m_tarBufferCPU_ICGN.reset(new TW::Concurrent_Buffer<cv::Mat>(20));
		}

		// Destroy the dialog
		deleteObject(m_camParamDialog);

		// Create the display widget
		if(ComputationMode::GPUFFTCC_CPUICGN == m_computationMode)
		{		
			m_onecamWidget = new OneCamWidget(
				0,
				m_refBuffer,
				m_tarBuffer,
				m_refBufferCPU_ICGN,
				m_tarBufferCPU_ICGN,
				iNumICGNThreads,
				m_iImgWidth,
				m_iImgHeight,
				iNumberX,
				iNumberY,
				m_ROI,
				m_computationMode,
				this);
		}
		else
		{
			m_onecamWidget = new OneCamWidget(
				0,
				m_refBuffer,
				m_tarBuffer,
				m_iImgWidth,
				m_iImgHeight,
				iNumberX,
				iNumberY,
				m_ROI,
				m_computationMode,
				this);
		}
		connect(m_onecamWidget, &OneCamWidget::titleReady, this, &TW_CoreMainWindow::updateTitle);

		// Try to connect to the camera
		if (m_onecamWidget->connectToCamera(
			m_isDropFrameChecked,
			m_iSubsetX, m_iSubsetY,
			m_iGridSpaceX, m_iGridSpaceY,
			m_iMarginX, m_iMarginY,
			m_ROI))
		{
			ui.gridLayout->addWidget(m_onecamWidget);
		}
	}
	else
	{
		deleteObject(m_camParamDialog);
	}
}

void TW_CoreMainWindow::updateTitle(const QString& qstr)
{
	setWindowTitle(QLatin1String("TW_Core_Application: ") + qstr);
}