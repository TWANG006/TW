#include "camparamdialog.h"

CamParamDialog::CamParamDialog(QWidget *parent)
	: QDialog(parent)
{
	ui.setupUi(this);
	ui.frameLable->setText(tr("No camera connected!"));

	ui.frameRateLabel->setText("");
	ui.cameraIDLabel->setText("");
	ui.cameraResolutionLabel->setText("");
	ui.mouseCursorLabel->setText("");
	ui.roiLabel->setText("");

	// Connect signals/slots
	connect(ui.frameLable, &FrameLabel::sig_mouseMove,
		    this		 , &CamParamDialog::updateMouseCursorPosLabel);

	qRegisterMetaType<ThreadStatisticsData>("ThreadStatisticsData");
}

CamParamDialog::~CamParamDialog()
{

}

bool CamParamDialog::connectToCamera(int width, int height)
{
	// Create the capture thread
	m_captureThread.reset(new CamParamThread(-1,-1));

	if(m_captureThread->connectToCamera())
	{
		connect(m_captureThread.data(), &CamParamThread::newFrame,
			    this,					&CamParamDialog::updateFrame);	

		m_captureThread->start();

		return true;
	}
	
	else
	{
		return false;
	}
}

void CamParamDialog::newMouseData(const MouseData& mouseData)
{

}

void CamParamDialog::updateFrame(const QImage &frame)
{
	// Display frame
	ui.frameLable->setPixmap(QPixmap::fromImage(frame).scaled(ui.frameLable->width(), 
															  ui.frameLable->height(), 
															  Qt::KeepAspectRatio));
}

void CamParamDialog::updateMouseCursorPosLabel()
{

}