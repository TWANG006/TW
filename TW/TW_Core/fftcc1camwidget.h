#ifndef FFTCC1CAMWIDGET_H
#define FFTCC1CAMWIDGET_H

#include <QWidget>
#include "ui_fftcc1camwidget.h"

class FFTCC1CamWidget : public QWidget
{
	Q_OBJECT

public:
	FFTCC1CamWidget(QWidget *parent = 0);
	~FFTCC1CamWidget();

private:
	Ui::FFTCC1CamWidget ui;
};

#endif // FFTCC1CAMWIDGET_H
