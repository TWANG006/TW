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

	clock.reset();

	const int NUM_TESTS = 1 + rand() % 10;
	const float THRESHOLD = 0.1f;

	for (int i = 0; i < NUM_TESTS; i++)
	{
		int thisTestTimeMilliseconds = rand() % 10000;
		float thisTestTimeSeconds = thisTestTimeMilliseconds / 1000.0f;
		clock.reset();
		QTest::qSleep(thisTestTimeMilliseconds);
		float elapsedSeconds = clock.getElapsedTime() / 1000;
		EXPECT_TRUE((thisTestTimeSeconds - THRESHOLD) < elapsedSeconds);
		EXPECT_TRUE(elapsedSeconds < (thisTestTimeSeconds + THRESHOLD));
	}
}