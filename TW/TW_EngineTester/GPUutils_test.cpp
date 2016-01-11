#include "gtest\gtest.h"
#include <Qdebug>

#include "TW.h"
#include "TW_utils.h"

using namespace TW;

TEST(POIpos1, POI_Position)
{
	int_t *h, *d;

	cuComputePOIPostions(
		d, h,
		4, 4,
		3, 3,
		16, 16,
		5, 5);

	for (auto i = 0; i < 4; i++)
	{

		for (auto j = 0; j < 4; j++)
		{
			qDebug() << h[(i * 4 + j)*2] << ", " << h[(i * 4 + j) * 2 + 1];
		}
		qDebug() << "\n";
	}

	cudaFree(d);
	free(h);
}