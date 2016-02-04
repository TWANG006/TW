#include "camparamdialog.h"

CamParamDialog::CamParamDialog(QWidget *parent)
	: QDialog(parent)
{
	ui.setupUi(this);
	ui.frameLable->setText(tr("No camera connected!"));

	ui.frameRateLabel->setText("");
	ui.cameraIDLabel->setText("");
	ui.cameraResolutionLabel->setText("");
	ui.mouseCursorLabel->setText("");
	ui.roiLabel->setText("");


}

CamParamDialog::~CamParamDialog()
{

}
