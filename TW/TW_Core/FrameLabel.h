#ifndef FRAMELABEL_H
#define FRAMELABEL_H

#include <QLabel>
#include <QPoint>
#include <QRect>

struct MouseData
{
	bool m_isLeftBtnReleased;
	bool m_isRightBtnReleased;
	QRect m_roiBox;
};

class FrameLabel :public QLabel
{
	Q_OBJECT

public:
	FrameLabel(QWidget *parent = 0);
	QPoint GetCursorPos();
	void SetCursorPos(const QPoint& point);

protected:
	void mouseMoveEvent(QMouseEvent*) Q_DECL_OVERRIDE;
	void mousePressEvent(QMouseEvent*) Q_DECL_OVERRIDE;
	void mouseReleaseEvent(QMouseEvent*) Q_DECL_OVERRIDE;
	void paintEvent(QPaintEvent*) Q_DECL_OVERRIDE;

private:
	QPoint m_starPoint;
	QPoint m_cursorPos;
	QScopedPointer<QRect> m_roiBox;
	MouseData m_mouseData;


signals:
	void sig_newROI(const MouseData& mouseData);
	void sig_mouseMove();
};


#endif // !FRAMELABLE_H
