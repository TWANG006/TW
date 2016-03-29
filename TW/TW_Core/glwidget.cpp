#include "glwidget.h"

GLWidget::GLWidget(QWidget *parent)
	: QOpenGLWidget(parent)
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

}

void GLWidget::resizeGL(int w, int h)
{
}