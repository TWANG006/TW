#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "TW.h"

/// \brief GPU Thrust-based min_max reduction function. The results are on GPU side
/// 
/// \param input the input raw device pointer holding the array needing min_max
/// \param min the raw device pointer output of min ele
/// \param max the raw device pointer output of max ele
/// \param n   the size of the input array
void minMaxRWrapper(TW::real_t *&iU, TW::real_t *&iV, int iNU, int iNV,
				    TW::real_t* &iminU, TW::real_t* &imaxU,
					TW::real_t* &iminV, TW::real_t* &imaxV);

void constructTextImage();

void cuUpdatePOIpos(// Inputs
					TW::real_t *fU,
				    TW::real_t *fV,
					int iNumberX, int iNumberY,
					// Outputs
					int *iPOIpos);

void cuAccumulatePOI_UV(// Inputs
						TW::real_t *fCurrentU,
						TW::real_t *fCurrentV,
						int *iCurrentPOIXY,
						int iNumPOI,
						// Outputs
						TW::real_t *fU,
						TW::real_t *fV,
						int *iPOIXY);


#endif // !CUDA_UTILS_CUH
