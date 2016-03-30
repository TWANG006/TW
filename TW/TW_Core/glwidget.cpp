#include "glwidget.h"

GLWidget::GLWidget(QWidget *parent,
				   QThread *&renderThread)
	: QOpenGLWidget(parent)
	, m_renderThread(renderThread)
{

}

GLWidget::~GLWidget()
{
	makeCurrent();

	doneCurrent();
}

void GLWidget::initializeGL()
{
	initializeOpenGLFunctions();
	glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
}

void GLWidget::paintGL()
{
	 makeCurrent();

	 

	 doneCurrent();
}

void GLWidget::resizeGL(int w, int h)
{
}