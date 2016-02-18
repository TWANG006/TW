#include "camparamdialog.h"
#include <QDebug>
#include <QMessageBox>

CamParamDialog::CamParamDialog(QWidget *parent)
	: QDialog(parent)
	, m_isCameraConnected(false)
	, m_captureThread(nullptr)
	, m_ROIRect()
{
	ui.setupUi(this);
	ui.frameLable->setText(tr("Connecting to camera..."));

	ui.frameRateLabel->setText("");
	ui.cameraIDLabel->setText("0");
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
	if(m_isCameraConnected)
	{
		// Stop the capture thread
		if(m_captureThread->isRunning())
			stopCaptureThread();

		// Disconnect the camera
		if(m_captureThread->disconnectCamera())
			qDebug() <<"Camera sucessfully disconnected.";
		else
			qDebug() <<"WARNING: Camera already disconnected.";
	}
}

bool CamParamDialog::connectToCamera(int width, int height)
{
	// Create the capture thread
	m_captureThread.reset(new CamParamThread(-1,-1, this));

	if(m_captureThread->connectToCamera())
	{
		// signal/slots connections between capture thread and camera dialog
		connect(m_captureThread.data(), &CamParamThread::newFrame,
			    this,					&CamParamDialog::updateFrame);	
		connect(this,                   &CamParamDialog::setROI,
			    m_captureThread.data(), &CamParamThread::setROI);
		connect(ui.frameLable,			&FrameLabel::sig_newMouseData,
			    this,					&CamParamDialog::newMouseData);

		// Set the initial ROI for the capture thread
		m_ROIRect.setRect(0,
						  0,
						  m_captureThread->getInputSourceWidth(),
						  m_captureThread->getInputSourceHeight());
		emit setROI(m_ROIRect);

		// Start capturing (the capture thread event loop begins here)
		m_captureThread->start();
		
		// Set the camera connected flag to true
		m_isCameraConnected = true;

		return true;
	}
	
	else
	{
		return false;
	}
}

void CamParamDialog::stopCaptureThread()
{
	qDebug() <<"Trying to stop capture thread...";
	m_captureThread->stop();
	
	if(m_captureThread->wait())
		qDebug()<<"Capture thread is stopped successfully.";
}

void CamParamDialog::newMouseData(const MouseData& mouseData)
{
	int x_temp, y_temp, width_temp, height_temp;
	QRect m_ROIRect;

	if(mouseData.m_isLeftBtnReleased)
	{
		double xScalingFactor;
		double yScalingFactor;
		double wScalingFactor;
		double hScalingFactor;

		// Compute the scaling factors for x&y axis
		// xScalingFactor = (ROI.x - (f.width-p.width)/2) / p.width
		xScalingFactor = ((double)mouseData.m_roiBox.x() - ((ui.frameLable->width() - ui.frameLable->pixmap()->width())/2)) / double(ui.frameLable->pixmap()->width());
		yScalingFactor = ((double)mouseData.m_roiBox.y() - ((ui.frameLable->height() - ui.frameLable->pixmap()->height())/2)) / double(ui.frameLable->pixmap()->height());
		wScalingFactor = (double)m_captureThread->GetCurrentROI().width() / (double)ui.frameLable->pixmap()->width();
		hScalingFactor = (double)m_captureThread->GetCurrentROI().height() / (double)ui.frameLable->pixmap()->height();

		m_ROIRect.setX(xScalingFactor*m_captureThread->GetCurrentROI().width() + m_captureThread->GetCurrentROI().x());
		m_ROIRect.setY(yScalingFactor*m_captureThread->GetCurrentROI().height() + m_captureThread->GetCurrentROI().y());
		m_ROIRect.setWidth(wScalingFactor*mouseData.m_roiBox.width());
		m_ROIRect.setHeight(hScalingFactor*mouseData.m_roiBox.height());

		// Check if selection box has NON-zero dimensions
		if((m_ROIRect.width() != 0) && (m_ROIRect.height() != 0))
		{
			// If the box is drawn from bottom-right to top-left, reverse
			// the coordinates
			if(m_ROIRect.width() < 0)
			{
				x_temp = m_ROIRect.x();
				width_temp = m_ROIRect.width();
				m_ROIRect.setX(x_temp + m_ROIRect.width());
				m_ROIRect.setWidth(width_temp * -1);
			}
			if(m_ROIRect.height() < 0)
			{
				y_temp = m_ROIRect.y();
				height_temp = m_ROIRect.height();
				m_ROIRect.setY(y_temp + m_ROIRect.height());
				m_ROIRect.setHeight(height_temp * -1);
			}

			// Make sure the box is not outside the window
			if((m_ROIRect.x()<0) || (m_ROIRect.y()<0) ||
				((m_ROIRect.x() + m_ROIRect.width())>(m_captureThread->GetCurrentROI().x() + m_captureThread->GetCurrentROI().width())) ||
				((m_ROIRect.y() + m_ROIRect.height())>(m_captureThread->GetCurrentROI().y() + m_captureThread->GetCurrentROI().height())) ||
				(m_ROIRect.x() < m_captureThread->GetCurrentROI().x()) ||
				(m_ROIRect.y() < m_captureThread->GetCurrentROI().y()))
			{
				QMessageBox::critical(this, 
									  "Invalid ROI",
									  tr("ROI is out of the boudary"));
			}

			// Send the reset ROI signal
			else
			{
				emit setROI(m_ROIRect);
			}
		}
	}

	if(mouseData.m_isRightBtnReleased)
	{
		m_ROIRect.setRect(0,
						  0,
						  m_captureThread->getInputSourceWidth(),
						  m_captureThread->getInputSourceHeight());
		emit setROI(m_ROIRect);
	}
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
	// Update mouse cursor position shown in the mouseCursorLabel
	ui.mouseCursorLabel->setText(QLatin1String("(") + QString::number(ui.frameLable->GetCursorPos().x()) + 
								 QLatin1String("(") + QString::number(ui.frameLable->GetCursorPos().y()) + 
								 QLatin1String(")"));

	// Show pixel cursor position if camera is connected
	if(ui.frameLable->pixmap() != 0)
	{
		
	}
}