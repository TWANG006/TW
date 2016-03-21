#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

/// \brief GPU Thrust-based min_max reduction function. The results are on GPU side
/// 
/// \param input the input raw device pointer holding the array needing min_max
/// \param min the raw device pointer output of min ele
/// \param max the raw device pointer output of max ele
/// \param n   the size of the input array
void minMaxRWrapper(int *iU, int *iV, int iNU, int iNV,
				    int* iminU, int* imaxU,
					int* iminV, int* imaxV);

void updatePOI_ROI(int *iPOIpos,
				   int *iU,
				   int *iV,
				   int iSubsetX,
				   int iSubsetY,
				   int iMarginX,
				   int iMarginY,
				   int &iStartX,
				   int &iStartY,
				   int &iROIWidth,
				   int &iROIHeight);


#endif // !CUDA_UTILS_CUH
