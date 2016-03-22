#include "gtest\gtest.h"
#include <Qdebug>

#include "TW.h"
#include "TW_utils.h"
#include "TW_cuTempUtils.h"
#include "GPUThrust.cuh"



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

	int m,n;

	maxReduction(d, m, n, 10);

	qDebug()<<m<<",  "<<n;

	delete h; h = nullptr;
	cudaFree(d);
}

TEST(THRUST_minmax, Thrust_minmax)
{
	int *h, *d1, *d2;
	int *h1r, *h1rr, *h2r, *h2rr, *d1r, *d1rr, *d2r, *d2rr;
	h = new int[10];
	h1r = new int[1];
	h2r = new int[1];
	h1rr = new int[1];
	h2rr = new int[1];

	for(int i=0;i<10;i++)
		h[i] = i;

	cudaMalloc((void**)&d1r, sizeof(int));
	cudaMalloc((void**)&d2r, sizeof(int));
	cudaMalloc((void**)&d1rr, sizeof(int));
	cudaMalloc((void**)&d2rr, sizeof(int));
	cudaMalloc((void**)&d1, sizeof(int)*10);
	cudaMalloc((void**)&d2, sizeof(int)*10);


	cudaMemcpy(d1,h,sizeof(int)*10,cudaMemcpyHostToDevice);
	cudaMemcpy(d2,h,sizeof(int)*10,cudaMemcpyHostToDevice);

	minMaxRWrapper(d1,d2,10,10,d1r,d1rr,d2r,d2rr);

	cudaMemcpy(h1r, d1r,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(h2r, d2r,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(h1rr,d1rr,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(h2rr,d2rr,sizeof(int),cudaMemcpyDeviceToHost);

	qDebug()<<*h1r<<","<<*h1rr;
	qDebug()<<*h2r<<","<<*h2rr;

	cudaFree(d1);
	cudaFree(d2);
	cudaFree(d1r);
	cudaFree(d2r);
	cudaFree(d1rr);
	cudaFree(d2rr);
	delete h; h = nullptr;
	delete h1r; h1r = nullptr;
	delete h2r; h2r = nullptr;
	delete h1rr; h1rr = nullptr;
	delete h2rr; h2rr = nullptr;
}