#include <gtest\gtest.h>

#include <QPoint>
#include <cuda_runtime.h>
#include <concurrent_queue.h>
#include <deque>
#include <QRect>
#include <QImage>

#include "TW_MatToQImage.h"

struct MouseData
{
	bool m_isLeftBtnReleased;
	bool m_isRightBtnReleased;
	QRect m_roiBox;
};


class dummy
{
public:
	dummy()
		: p(2,2)
	{}

	int GetX() const { return p.x();}

private:
	QPoint p;
	MouseData m;
};

TEST(cudaFree, isEqualtoNullorNot)
{

	dummy d;

	EXPECT_EQ(d.GetX(), 2);

	cv::Mat m;

	QImage q = TW::Mat2QImage(m);

	/*int *x;
	cudaMalloc((void**)&x, sizeof(int) * 10);

	std::cout<<sizeof(x)/sizeof(x[0])<<std::endl;

	cudaFree(x);
	x = nullptr;

	concurrency::concurrent_queue<int>* q;

	EXPECT_EQ(x, nullptr);*/
}