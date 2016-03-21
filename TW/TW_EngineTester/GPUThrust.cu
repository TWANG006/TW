#include "thrust\reduce.h"
#include "thrust\device_ptr.h"
#include "thrust\count.h"
#include "thrust\extrema.h"
#include "thrust\detail\static_assert.h"
#include "thrust\device_free.h"
#include <QDebug>

void maxReduction(int* i, int& o, int& o2, int n)
{
	thrust::device_ptr<int> dev_p(i);
	thrust::pair<thrust::device_ptr<int>, thrust::device_ptr<int>> max = thrust::minmax_element(dev_p, dev_p+n);

	o = *max.first;
	o2 = *max.second;
}

void memtest()
{
	int *h, *d;
	
	h = new int[10];

	for(int i=0;i<10;i++)
		h[i] = i;

	cudaMalloc((void**)&d, sizeof(int)*10);
	cudaMemcpy(d,h,sizeof(int)*10,cudaMemcpyHostToDevice);

	thrust::device_ptr<int> d_ptr(d);

	thrust::device_ptr<int> result = thrust::max_element(d_ptr, d_ptr+10);



	cudaMemcpy(h,d,sizeof(int)*10,cudaMemcpyHostToDevice);

	qDebug()<<h[8];

	
	
	qDebug()<<result[0];
	cudaFree(d); 
	
	delete h; h=nullptr;
}