#ifndef FFTCC1CAMWIDGET_H
#define FFTCC1CAMWIDGET_H

#include <QWidget>
#include "ui_fftcc1camwidget.h"

#include "Structures.h"

class FFTCC1CamWidget : public QWidget
{
	Q_OBJECT

public:
	FFTCC1CamWidget(int deviceNumber,
					ImageBufferPtr refImgBuffer,
					ImageBufferPtr tarImgBuffer,
					QWidget *parent = 0);
	~FFTCC1CamWidget();

public slots:
	void updateRefFrame(const QImage&);
	void updateTarFrame(const QImage&);

private:
	Ui::FFTCC1CamWidget ui;

	int m_iDeviceNumber;
	ImageBufferPtr m_refImgBuffer;
	ImageBufferPtr m_tarImgBuffer;
};

#endif // FFTCC1CAMWIDGET_H
