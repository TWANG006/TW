//#include "TW_MemManager.h"
//
//#include <gtest\gtest.h>
//
//using namespace TW;
//
//TEST(hcreateptr, Host_memory_allocation)
//{
//	float *ptr;
//	hcreateptr<float>(ptr, 4);
//
//	ptr[3] =12;
//	std::cout<<ptr[3]<<std::endl;
//
//	hdestroyptr<float>(ptr);
//	EXPECT_EQ(nullptr, ptr);
//}
//
//TEST(cucreatptr, Pinned_memory_allocation)
//{
//	float ****ptr;
//	cucreateptr<float>(ptr, 1, 2, 3, 4);
//	cudestroyptr<float>(ptr);
//	EXPECT_EQ(nullptr, ptr);
//}
