#include "FFTCC.h"

#include <gtest\gtest.h>

using namespace TW::Algorithm;

TEST(Fftcc, Constructor)
{
	Fftcc * fftcc = new Fftcc(
		100,
		100);

	EXPECT_EQ(fftcc->getNumPOIsX(), fftcc->getNumPOIsY());

	delete fftcc;
	fftcc = nullptr;
}