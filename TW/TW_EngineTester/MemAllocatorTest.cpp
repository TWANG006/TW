//#include "MemManager.h"
//
//#include <gtest\gtest.h>
//
//using namespace TW::MemManager;
//
//TEST(hcreateptr, Host_memory_allocation)
//{
//	float *ptr;
//	hcreateptr<float>(ptr, 4);
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
