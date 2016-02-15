#include "FrameLabel.h"

#include <QPainter>
#include <QDebug>

FrameLabel::FrameLabel(QWidget *parent)
	: QLabel(parent)
	, m_starPoint(0,0)
	, m_cursorPos(0,0)
	, m_isDrawingBox(false)
{
	m_mouseData.m_isLeftBtnReleased = false;
	m_mouseData.m_isRightBtnReleased = false;
	// createContextMenu();
}

QPoint FrameLabel::GetCursorPos() const
{
	return m_cursorPos;
}

void FrameLabel::SetCursorPos(const QPoint& point)
{
	m_cursorPos = point;
}

void FrameLabel::mouseMoveEvent(QMouseEvent* evt)
{
	// Update mouse cursor position
	SetCursorPos(evt->pos());

	// Update the ROI width and height 
	if(m_isDrawingBox)
	{
		m_roiBox->setWidth(GetCursorPos().x() - m_starPoint.x());
		m_roiBox->setHeight(GetCursorPos().y() - m_starPoint.y());
	}

	// Inform main window the mouse is moving
	emit sig_mouseMove();
}

void FrameLabel::mousePressEvent(QMouseEvent* evt)
{
	// Update the cursor position
	SetCursorPos(evt->pos());

	// If Left button is pressed, begin to draw the ROI
	if(evt->button() == Qt::LeftButton)
	{
		// Start drawing box
		m_starPoint = evt->pos();
		m_roiBox.reset(new QRect(m_starPoint.x(), m_starPoint.y(), 0,0));
		m_isDrawingBox = true;
	}
}

void FrameLabel::mouseReleaseEvent(QMouseEvent* evt)
{
	// Update cursor position
	SetCursorPos(evt->pos());

	// LeftButton is released, stop drawing ROI, and save the new ROI
	// result
	if(evt->button() == Qt::LeftButton)
	{
		m_mouseData.m_isLeftBtnReleased = true;
		
		if(m_isDrawingBox)
		{
			// Stop drawing ROI
			m_isDrawingBox = false;
			
			// Update the ROI
			m_mouseData.m_roiBox.setX(m_roiBox->left());
			m_mouseData.m_roiBox.setY(m_roiBox->top());
			m_mouseData.m_roiBox.setWidth(m_roiBox->width());
			m_mouseData.m_roiBox.setHeight(m_roiBox->height());

			m_mouseData.m_isLeftBtnReleased = true;

			emit sig_newMouseData(m_mouseData);
		}

		m_mouseData.m_isLeftBtnReleased = false;
	}

	else if (evt->button() == Qt::RightButton)
	{
		if(m_isDrawingBox)
			m_isDrawingBox = false;
		else
			emit sig_resetROI();
	}
}

void FrameLabel::paintEvent(QPaintEvent* evt)
{
	QLabel::paintEvent(evt);
	QPainter painter(this);

	if(m_isDrawingBox)
	{
		painter.setPen(Qt::red);
		painter.drawRect(*m_roiBox.data());
	}

	update();
}

//void FrameLabel::createContextMenu()
//{
//	// Create the menu object
//	m_menu.reset(new QMenu(this));
//
//	// Add action entries to the context menu
//	QScopedPointer<QAction> action(new QAction(this));
//	action->setText(tr("Reset ROI"));	
//	m_menu->addAction(action.take());
//
//	action.reset(new QAction(this));
//	action->setText(tr("GrayScale"));
//}