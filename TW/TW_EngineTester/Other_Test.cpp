#include <gtest\gtest.h>
#include <QtConcurrent\QtConcurrentRun>
#include <QThread>
#include <vector>
#include <functional>
#include <iostream>

using namespace std;

void add1(int *a, int start, int end)
{
	for(int i=start; i<end; i++)
	{
		a[i] += 1;
	}
	float i=0;
	while (i<100000)
	{
		i+=0.5;
	}
	cout<<i<<end;
	cout<<"Thread ID: "<<QThread::currentThreadId()<<endl;
}

TEST(QtConccurentRun_for_Pointers, QCRP)
{
	int *a = new int[10]{0};

	vector<QFuture<void>> v;

	for(int i=0; i<100; i++)
	{
		v.push_back(QtConcurrent::run(add1,a,0,10));
	}

	for_each(v.begin(), v.end(), mem_fn(&QFuture<void>::waitForFinished));

	/*QFuture<void> f = QtConcurrent::run(add1, a, 0, 6);
	f.waitForFinished();
	QFuture<void> f1 = QtConcurrent::run(add1, a, 6, 10);
	f1.waitForFinished();*/

	for(int i=0; i<10; i++)
	{
		cout<<a[i]<<", ";
	}

	delete [] a;
}
//
//#include <QPoint>
//#include <cuda_runtime.h>
//#include <concurrent_queue.h>
//#include <deque>
//#include <QRect>
//#include <QImage>
//
//#include "TW_MatToQImage.h"
//
//struct MouseData
//{
//	bool m_isLeftBtnReleased;
//	bool m_isRightBtnReleased;
//	QRect m_roiBox;
//};
//
//
//class dummy
//{
//public:
//	dummy()
//		: p(2,2)
//	{}
//
//	int GetX() const { return p.x();}
//
//private:
//	QPoint p;
//	MouseData m;
//};
//
//TEST(cudaFree, isEqualtoNullorNot)
//{
//
//	dummy d;
//
//	EXPECT_EQ(d.GetX(), 2);
//
//	cv::Mat m;
//
//	QImage q = TW::Mat2QImage(m);
//
//	/*int *x;
//	cudaMalloc((void**)&x, sizeof(int) * 10);
//
//	std::cout<<sizeof(x)/sizeof(x[0])<<std::endl;
//
//	cudaFree(x);
//	x = nullptr;
//
//	concurrency::concurrent_queue<int>* q;
//
//	EXPECT_EQ(x, nullptr);*/
//}