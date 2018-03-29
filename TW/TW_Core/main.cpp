#include "tw_coremainwindow.h"
#include <QtWidgets/QApplication>
#include <QTranslator>
#include <cuda.h>

int main(int argc, char *argv[])
{
	qRegisterMetaType<QVector<float> >("QVector<float>");

	QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);

	QApplication a(argc, argv);
	
	QTranslator translator;

	if(QLocale::system().name().contains("zh"))
	{
		translator.load("tw_core_zh.qm");
		a.installTranslator(&translator);
	}
	
	TW_CoreMainWindow w;
	w.show();

	cudaSetDevice(0);

	return a.exec();
}