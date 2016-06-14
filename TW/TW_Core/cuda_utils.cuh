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

void constructTextImage(// Outputs
						unsigned int* texImgU,
						unsigned int* texImgV,
						// Inputs
						int *iPOIpos,
						TW::real_t* fU,
						TW::real_t* fV,
						int iNumPOI,
						int iStartX, int iStartY,
						int iROIWidth, int iROIHeight,
						TW::real_t* fMaxU, TW::real_t* fMinU,
						TW::real_t* fMaxV, TW::real_t* fMinV);

void constructTextImageFixedMinMax(// Outputs
						unsigned int* texImgU,
						unsigned int* texImgV,
						// Inputs
						int *iPOIpos,
						TW::real_t* fU,
						TW::real_t* fV,
						TW::real_t* fAccumulateU,
						TW::real_t* fAccumulateV,
						int iNumPOI,
						int iStartX, int iStartY,
						int iROIWidth, int iROIHeight,
						TW::real_t fMaxU, TW::real_t fMinU,
						TW::real_t fMaxV, TW::real_t fMinV);

void cuUpdatePOIpos(// Inputs
					TW::real_t *fU,
				    TW::real_t *fV,
					int iNumberX, int iNumberY,
					// Outputs
					int *iPOIpos);

void cuAccumulatePOI(// Inputs
						TW::real_t *fCurrentU,
						TW::real_t *fCurrentV,
						int *iCurrentPOIXY,
						int iNumPOI,
						// Outputs
						int *iPOIXY);

void cuAccumulatePOI(// Inputs
						TW::real_t *fU,
						TW::real_t *fV,
						TW::real_t *fAccumulateU,
						TW::real_t *fAccumulateV,
						int *iCurrentPOIXY,
						int iNumPOI,
						// Outputs
						int *iPOIXY);

void cuAccumulateUV(// Inputs
									TW::real_t *fCurrentU,
									TW::real_t *fCurrentV,
									int iPOINum,
									// Outputs
									TW::real_t *fU,
									TW::real_t *fV);

#endif // !CUDA_UTILS_CUH
