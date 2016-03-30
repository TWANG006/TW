#include "Structures.h"

// Allocating and intializing the singleton class' static data member. 
// Note: the pointer is initialized, not the object itself
SharedResources *SharedResources::g_instance = nullptr;

