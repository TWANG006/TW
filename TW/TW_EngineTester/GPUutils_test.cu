#include "gtest\gtest.h"
#include <Qdebug>

#include "TW.h"
#include "TW_utils.h"
#include "TW_cuTempUtils.h"
#include "thrust\reduce.h"
#include "thrust\device_ptr.h"
#include "thrust\count.h"
#include "thrust\extrema.h"
#include "thrust\detail\static_assert.h"

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

TEST(CUDAInitialization, cuInitialize)
{
	int *h, *d;
	h = new int[10];
	cudaMalloc((void**)&d, sizeof(int)*10);

	cuInitialize<int>(d, 10, sizeof(int)*10,0);

	cudaMemcpy(h,d,sizeof(int)*10,cudaMemcpyDeviceToHost);

	EXPECT_EQ(10, h[0]);
	EXPECT_EQ(10, h[9]);

	qDebug()<<h[0]<<h[3];

	cudaFree(d);
	delete h; h= nullptr;
}

TEST(THRUST, deviceReduction)
{
	int *h, *d;
	
	h = new int[10];

	for(int i=0;i<10;i++)
		h[i] = i;

	cudaMalloc((void**)&d, sizeof(int)*10);
	cudaMemcpy(d,h,sizeof(int)*10,cudaMemcpyHostToDevice);

	thrust::device_ptr<int> dev_p(d);
	thrust::device_ptr<int> max = thrust::max_element(dev_p, dev_p+10);

	int m = max[0];

	qDebug()<<m;
}