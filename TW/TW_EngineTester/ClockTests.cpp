#include <gtest\gtest.h>
#include "Clock.h"
#include <QTest>

using namespace TW::Timing;

TEST(CClock, Initialize)
{
	CClock clock;
	EXPECT_TRUE(clock.initialize());
	EXPECT_TRUE(clock.shutdown());
}

TEST(CClock, FrameTimeMeasuing)
{
	CClock clock;

	EXPECT_TRUE(clock.initialize());
	QTest::qSleep(1000);
	clock.newFrame();
	float timedTime = clock.timeElapsedLastFrame();
	EXPECT_TRUE(0.9f < timedTime);
	EXPECT_TRUE(timedTime < 1.1f);
	clock.newFrame();
	QTest::qSleep(500);
	clock.newFrame();
	timedTime = clock.timeElapsedLastFrame();
	EXPECT_TRUE(0.4f < timedTime);
	EXPECT_TRUE(timedTime < 0.6f);

	const int NUM_TESTS = 10 + rand() % 100;
	const float THRESHOLD = 0.1f;

	for (int i = 0; i < NUM_TESTS; i++)
	{
		int thisTestTimeMilliseconds = rand() % 10000;
		float thisTestTimeSeconds = thisTestTimeMilliseconds / 1000.0f;
		clock.newFrame();
		QTest::qSleep(thisTestTimeMilliseconds);
		clock.newFrame();
		float elapsedSeconds = clock.timeElapsedLastFrame();
		EXPECT_TRUE((thisTestTimeSeconds - THRESHOLD) < elapsedSeconds);
		EXPECT_TRUE(elapsedSeconds < (thisTestTimeSeconds - THRESHOLD));
	}
	
	clock.newFrame();
	clock.timeElapsedLastFrame();
	EXPECT_TRUE(clock.shutdown());
}