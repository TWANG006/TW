#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QThread>
#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLVertexArrayObject>
#include "Structures.h"

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core
{
	Q_OBJECT

public:
	GLWidget(SharedResources*&, QThread *&, QWidget *parent);
	~GLWidget();

protected:
	void initializeGL() Q_DECL_OVERRIDE;
	void paintGL() Q_DECL_OVERRIDE;
	void resizeGL(int w, int h) Q_DECL_OVERRIDE;

private:
	SharedResources *m_sharedResources;
	QThread *m_renderThread;
	QOpenGLVertexArrayObject m_vao;

	GLint m_viewWidth;
	GLint m_viewHeight;
};

#endif // GLWIDGET_H
