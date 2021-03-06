#ifndef TW_H
#define TW_H

#define TW_PI 3.14159265358979323846
#define TW_TWOPI 6.28318530717958647692

#define BLOCK_SIZE_256 256
#define BLOCK_SIZE_128 128
#define BLOCK_SIZE_64 64

//!- Macro for library dll export utility
#if defined (_WIN32)
#    if defined (TW_LIB_DLL_EXPORTS_MODE)
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
#include <fftw3.h>

//#define TW_USE_DOUBLE
namespace TW
{
	//!- TW basic types
#ifdef TW_USE_DOUBLE
using intentisy_t = double;
using real_t = double;
using real_t2 = double2;
using real_t4 = double4;
using cudafftComplex = cufftDoubleComplex;
using fftw3Plan = fftw_plan;
using fftw3Complex = fftw_complex;
#else
using intensity_t = float;
using real_t = float;
using real_t2 = float2;
using real_t4 = float4;
using cudafftComplex = cufftComplex;
using fftw3Plan = fftwf_plan;
using fftw3Complex = fftwf_complex;
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
