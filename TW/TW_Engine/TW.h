#ifndef TW_H
#define TW_H

#define TW_PI 3.14159265358979323846
#define TW_TWOPI 6.28318530717958647692

#define BLOCK_SIZE_256 256
#define BLOCK_SIZE_128 128

//!- Macro for library dll export utility
#ifdef _WIN32
#    ifdef TW_LIB_DLL_EXPORTS_MODE
#        define TW_LIB_DLL_EXPORTS __declspec(dllexport)
#    else
#        define TW_LIB_DLL_EXPORTS __declspec(dllimport)
#    endif
#else
#    define TW_LIB_DLL_EXPORTS
#endif

//!- TW engine version
#define VERSION "1.0"

//!- Debugging macro
#ifdef TW_DEBUG_MSG
#define DEBUG_MSG(x) do {std::cout<<"[TW_DEBUG]: "<<x<<std::endl;} while(0)
#else
#define DEBUG_MSG(x) do {} while(0);
#endif // TW_DEBUG_MSG



#include <string>
#include <cufft.h>
#include <cfloat>

namespace TW
{
	//!- TW basic types
#ifdef TW_USE_DOUBLE
using intentisy_t = double;
using real_t = double;
using cudafftComplex = cufftDoubleComplex;
#else
using intensity_t = float;
using real_t = float;
using cudafftComplex = cufftComplex;
#endif // TW_USE_DOUBLE
using int_t = int;
using uint_t = unsigned int;

// !- Setup whether to use CUDA, multicore or single core
enum PARALLEL_COMPUTING_TYPE
{
	Singlecore
	, Multicore
	, CUDA_GPU
	//, OpenCL
};

} //!- namespace TW


#endif // !TW_H
