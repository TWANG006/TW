#include <gtest\gtest.h>
#include "StopWatch.h"
#include <QTest>

using namespace TW::Timing;

TEST(StopWatch, FrameTimeMeasuing)
{
	StopWatch clock;

	clock.start();
	QTest::qSleep(1000);
	float timedTime = clock.getElapsedTime() / 1000;
	EXPECT_TRUE(0.9f < timedTime);
	EXPECT_TRUE(timedTime < 1.1f);
	clock.reset();
	QTest::qSleep(500);
	timedTime = clock.getElapsedTime() / 1000;
	EXPECT_TRUE(0.4f < timedTime);
	EXPECT_TRUE(timedTime < 0.6f);

	clock.stop();
}