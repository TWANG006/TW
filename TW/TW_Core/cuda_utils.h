#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

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

#endif // !CUDA_UTILS_H
