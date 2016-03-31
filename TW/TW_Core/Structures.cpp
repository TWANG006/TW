#include "Structures.h"

// Allocating and intializing the singleton class' static data member. 
// Note: the pointer is initialized, not the object itself
std::unique_ptr<SharedResources> SharedResources::g_instance;
std::once_flag SharedResources::m_onceFlag;

SharedResources& SharedResources::GetIntance()
{
	std::call_once(m_onceFlag,
		[]{
			g_instance.reset(new SharedResources);
	});

	return *g_instance.get();
}

