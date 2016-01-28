#ifndef CAMPARAMDIALOG_H
#define CAMPARAMDIALOG_H

#include <QDialog>
#include "ui_camparamdialog.h"

class CamParamDialog : public QDialog
{
	Q_OBJECT

public:
	CamParamDialog(QWidget *parent = 0);
	~CamParamDialog();

private:
	Ui::CamParamDialog ui;
};

#endif // CAMPARAMDIALOG_H
