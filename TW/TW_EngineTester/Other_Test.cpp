#include <gtest\gtest.h>

#include <cuda_runtime.h>

TEST(cudaFree, isEqualtoNullorNot)
{
	int *x;
	cudaMalloc((void**)&x, sizeof(int) * 10);

	std::cout<<sizeof(x)/sizeof(x[0])<<std::endl;

	cudaFree(x);
	x = nullptr;

	EXPECT_EQ(x, nullptr);
}