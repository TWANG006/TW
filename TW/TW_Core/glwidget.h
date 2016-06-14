#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QThread>
#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLVertexArrayObject>
#include <QRect>
#include "Structures.h"

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core
{
	Q_OBJECT

public:
	GLWidget(std::shared_ptr<SharedResources>&, 
			 QThread *&, 
			 QWidget *parent,
			 int iWidth,
			 int iHeight,
			 const QRect &roi);
	~GLWidget();

protected:
	void initializeGL() Q_DECL_OVERRIDE;
	void paintGL() Q_DECL_OVERRIDE;
	void resizeGL(int w, int h) Q_DECL_OVERRIDE;

	void initCUDAArray();
	void initGLTexture();

private:
	std::shared_ptr<SharedResources> m_sharedResources;
	QThread *m_renderThread;
	QOpenGLVertexArrayObject m_vao;
	QOpenGLVertexArrayObject m_ROIvao;

	QRect m_ROI;

	int m_iImgWidth;
	int m_iImgHeight;

	GLint m_viewWidth;
	GLint m_viewHeight;
};

#endif // GLWIDGET_H
