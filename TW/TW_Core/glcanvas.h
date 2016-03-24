#ifndef GLCANVAS_H
#define GLCANVAS_H

#include <QOpenGLWidget>
#include <QThread>
#include <QRect>
#include "ui_glcanvas.h"

#include "Structures.h"
#include "capturethread.h"
#include "fftcctworkerthread.h"


class GLCanvas : public QOpenGLWidget
{
	Q_OBJECT

public:
	GLCanvas(int iDeviceNumber,
			 bool isDropFrameEnabled,
			 int iImgWidth, int iImgHeight,
			 int iSubsetX, int iSubsetY,
			 int iGridSpaceX, int iGridSapceY,
			 int iMarginX, int iMarginY,
			 ImageBufferPtr refImgBuffer,
			 ImageBufferPtr tarImgBuffer,
			 const QRect& roi,
			 QWidget *parent = 0);
	~GLCanvas();

signals:
	void renderRequest();

private:
	Ui::GLCanvas ui;

	ImageBufferPtr m_refImgBuffer;
	ImageBufferPtr m_tarImgBuffer;
    bool m_isCameraConnected;

	QThread *m_fftccWorkerThread;
	CaptureThread *m_captureThread;
	FFTCCTWorkerThread *m_fftccWorker;
};

#endif // GLCANVAS_H
