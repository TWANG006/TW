#ifndef TW_ICGN2D_H
#define TW_ICGN2D_H

#include "TW.h"

namespace TW{
/// \brief class ICGN
/// This class implement the general CPU-based ICGN algorithm. The computation
/// unit is a subset. This class can be used as the basic class for multi-core
/// processing when used in paDIC algorithm.
class TW_LIB_DLL_EXPORTS ICGN2D
{
public:
	ICGN2D();
	~ICGN2D();

private:

};


} //!- namespace TW

#endif // !TW_ICGN2D_H
