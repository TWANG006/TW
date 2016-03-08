#include "gtest\gtest.h"
#include <Qdebug>

#include "TW.h"
#include "TW_utils.h"
//
using namespace TW;
//
//TEST(POIpos1, POI_Position)
//{
//	int_t *h, *d;
//
//	cuComputePOIPostions(
//		d, h,
//		157, 157,
//		5, 5,
//		16, 16,
//		3, 3);
//
//	for (auto i = 0; i < 4; i++)
//	{
//
//		for (auto j = 0; j < 4; j++)
//		{
//			std::cout<<" [ "<< h[(i * 4 + j)*2] << ", " << h[(i * 4 + j) * 2 + 1]<<"]";
//		}
//		std::cout << "\n";
//	}
//
//	cudaFree(d);
//	free(h);
//}

TEST(SAXPY_1, saxpy)
{
	float a[] = {1,1,1,1,1};
	float *b = new float[5];

	float *aa;
	cudaMalloc((void**)&aa, sizeof(float)*5);
	cudaMemcpy(aa,a,sizeof(float)*5, cudaMemcpyHostToDevice);

	cuSaxpy(5, 1, aa, 0, aa);

	cudaMemcpy(b,aa,sizeof(float)*5, cudaMemcpyDeviceToHost);

	qDebug()<<b[0]<<b[1]<<b[2]<<b[3]<<b[4];

	delete b;
	cudaFree(aa);
}