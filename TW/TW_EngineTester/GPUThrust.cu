//#include "thrust\reduce.h"
//#include "thrust\device_ptr.h"
//#include "thrust\count.h"
//#include "thrust\extrema.h"
//#include "thrust\detail\static_assert.h"
//#include "thrust\device_free.h"
//#include <QDebug>
//
//void maxReduction(int* i, int& o, int& o2, int n)
//{
//	thrust::device_ptr<int> dev_p(i);
//	thrust::pair<thrust::device_ptr<int>, thrust::device_ptr<int>> max = thrust::minmax_element(dev_p, dev_p+n);
//
//	o = *max.first;
//	o2 = *max.second;
//}
//
//void memtest()
//{
//	int *h, *d;
//	
//	h = new int[10];
//
//	for(int i=0;i<10;i++)
//		h[i] = i;
//
//	cudaMalloc((void**)&d, sizeof(int)*10);
//	cudaMemcpy(d,h,sizeof(int)*10,cudaMemcpyHostToDevice);
//
//	thrust::device_ptr<int> d_ptr(d);
//
//	thrust::device_ptr<int> result = thrust::max_element(d_ptr, d_ptr+10);
//
//
//
//	cudaMemcpy(h,d,sizeof(int)*10,cudaMemcpyHostToDevice);
//
//	qDebug()<<h[8];
//
//	
//	
//	qDebug()<<result[0];
//	cudaFree(d); 
//	
//	delete h; h=nullptr;
//}
//
//void minMaxRWrapper(int *&iU, int *&iV, int iNU, int iNV,
//				    int* &iminU, int* &imaxU,
//					int* &iminV, int* &imaxV)
//{
//	using iThDevPtr = thrust::device_ptr<int>;
//
//	// Use thrust to find max and min simultaneously
//	iThDevPtr d_Uptr(iU);
//	thrust::pair<iThDevPtr, iThDevPtr> result_u = thrust::minmax_element(d_Uptr, d_Uptr+iNU);
//	// Cast the thrust device pointer to raw device pointer
//	iminU = thrust::raw_pointer_cast(result_u.first);
//	imaxU = thrust::raw_pointer_cast(result_u.second);
//
//	// Same for iV
//	iThDevPtr d_Vptr(iV);
//	thrust::pair<iThDevPtr, iThDevPtr> result_v = thrust::minmax_element(d_Vptr, d_Vptr+iNV);
//	// Cast the thrust device pointer to raw device pointer
//	iminV = thrust::raw_pointer_cast(result_u.first);
//	imaxV = thrust::raw_pointer_cast(result_u.second);
//
//	qDebug()<<result_u.first[0]<<", "<<result_u.second[0];
//}