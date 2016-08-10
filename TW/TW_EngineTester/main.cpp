#include <gtest\gtest.h>
#include "TW_paDIC_cuFFTCC2D.h"
#include "TW_paDIC_FFTCC2D_CPU.h"
#include "TW_utils.h"
#include "TW_MemManager.h"
#include <QCoreApplication>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>

using namespace TW;

int main(int argc, char *argv[])
{
	QCoreApplication app(argc, argv);

	::testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();


	return 0;
}
