#include "fftcc1camwidget.h"

FFTCC1CamWidget::FFTCC1CamWidget(int deviceNumber,
								 ImageBufferPtr refImgBuffer,
								 ImageBufferPtr tarImgBuffer,
								 QWidget *parent)
	: 
	, QWidget(parent)
{
	ui.setupUi(this);
	
}

FFTCC1CamWidget::~FFTCC1CamWidget()
{
	
}

void FFTCC1CamWidget::updateRefFrame(const QImage& refImg)
{
	ui.refFramelabel->setPixmap(QPixmap::fromImage(refImg).scaled(ui.refFramelabel->width(), 
															      ui.refFramelabel->height(), 
															      Qt::KeepAspectRatio));
}

void FFTCC1CamWidget::updateTarFrame(const QImage& tarImg)
{
	ui.tarFramelabel->setPixmap(QPixmap::fromImage(tarImg).scaled(ui.tarFramelabel->width(), 
															      ui.tarFramelabel->height(), 
															      Qt::KeepAspectRatio));
}