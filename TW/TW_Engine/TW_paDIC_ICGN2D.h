#ifndef TW_PADIC_ICGN2D_H
#define TW_PADIC_ICGN2D_H

#include <TW.h>
#include <opencv2\opencv.hpp>

namespace TW{
namespace paDIC{

/// \brief class ICGN
/// This class implement the general CPU-based ICGN algorithm. The computation
/// unit is based on two entire images. 
///	This class can be used as the basic class for multi-core
/// processing when used in paDIC algorithm.
class TW_LIB_DLL_EXPORTS ICGN2D
{
public:
	ICGN2D(const cv::Mat& refImg, 
		   const cv::Mat& tarImg
		   );
	virtual ~ICGN2D();

private:

};

} //!- namespace paDIC
} //!- namespace TW

#endif // !TW_PADIC_ICGN2D_H
