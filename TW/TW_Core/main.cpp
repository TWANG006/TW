#include "tw_coremainwindow.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	TW_CoreMainWindow w;
	w.show();
	return a.exec();
}