#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_2_Core>

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_2_Core
{
	Q_OBJECT

public:
	GLWidget(QWidget *parent);
	~GLWidget();

protected:
	void initializeGL() Q_DECL_OVERRIDE;
	void paintGL() Q_DECL_OVERRIDE;
	void resizeGL(int w, int h) Q_DECL_OVERRIDE;
private:
	
};

#endif // GLWIDGET_H
