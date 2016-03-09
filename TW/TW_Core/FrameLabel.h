#ifndef FRAMELABEL_H
#define FRAMELABEL_H

#include <QPointer>
#include <QLabel>
#include <QPoint>
#include <QRect>
#include <QMenu>
#include <QMouseEvent>

#include "Structures.h"


class FrameLabel :public QLabel
{
	Q_OBJECT

public:
	FrameLabel(QWidget *parent = 0);
	QPoint GetCursorPos() const;
	void SetCursorPos(const QPoint& point);

protected:
	void mouseMoveEvent(QMouseEvent*);
	void mousePressEvent(QMouseEvent*);
	void mouseReleaseEvent(QMouseEvent*);
	void paintEvent(QPaintEvent*);
	// void createContextMenu();

private:
	QPoint m_starPoint;
	QPoint m_cursorPos;
	QPointer<QRect> m_roiBox;
	MouseData m_mouseData;
	bool m_isDrawingBox;

signals:
	void sig_newMouseData(const MouseData& mouseData);
	void sig_mouseMove();
};


#endif // !FRAMELABLE_H
