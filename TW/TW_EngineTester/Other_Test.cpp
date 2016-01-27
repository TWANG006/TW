#include <gtest\gtest.h>

#include <cuda_runtime.h>
#include <concurrent_queue.h>
#include <deque>

TEST(cudaFree, isEqualtoNullorNot)
{
	int *x;
	cudaMalloc((void**)&x, sizeof(int) * 10);

	std::cout<<sizeof(x)/sizeof(x[0])<<std::endl;

	cudaFree(x);
	x = nullptr;

	concurrency::concurrent_queue<int>* q;

	EXPECT_EQ(x, nullptr);
}