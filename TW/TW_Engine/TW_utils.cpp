#include "TW_utils.h"
#include <iostream>

namespace TW
{
void ComputePOIPositions_s(// Output
					   	   int_t *&Out_h_iPXY,			// Return the host handle
						   // Inputs
						   int_t iNumberX, int_t iNumberY,
						   int_t iMarginX, int_t iMarginY,
						   int_t iSubsetX, int_t iSubsetY,
						   int_t iGridSpaceX, int_t iGridSpaceY)
{
	// TODO
}

void ComputePOIPositions_m(// Output
						   int_t *&Out_h_iPXY,			// Return the host handle
						   // Inputs
						   int_t iNumberX, int_t iNumberY,
						   int_t iMarginX, int_t iMarginY,
						   int_t iSubsetX, int_t iSubsetY,
						   int_t iGridSpaceX, int_t iGridSpaceY)
{
	// TODO
}

void Gradient_s(//Inputs
    			const cv::Mat& image,
				int_t iStartX, int_t iStartY,
				int_t iROIWidth, int_t iROIHeight,
				int_t iImgWidth, int_t iImgHeight,
				AccuracyOrder accuracyOrder,
				//Output
				real_t *Gx,
				real_t *Gy,
				real_t *Gxy)
{
	int_t iMarginX = iImgWidth - iROIWidth + 1;
	int_t iMarginY = iImgHeight - iROIHeight + 1;

	switch (accuracyOrder)
	{
	case TW::Quadratic:
	default:
		if(iStartX<1 || iStartY<1 || iMarginX<1 || iMarginY <1)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			if(Gx != nullptr)
				;
			if(Gy != nullptr)
				;
			if(Gxy!= nullptr)
				;
		}

		break;
	case TW::Quartic:
		if(iStartX<2 || iStartY<2 || iMarginX<2 || iMarginY <2)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		break;
	case TW::Octic:
		if(iStartX<4 || iStartY<4 || iMarginX<4 || iMarginY <4)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		break;
	}
}

void Gradient_m(//Inputs
				const cv::Mat& image,
				int_t iStartX, int_t iStartY,
				int_t iROIWidth, int_t iROIHeight,
				int_t iImgWidth, int_t iImgHeight,
				AccuracyOrder accuracyOrder,
				//Output
				real_t *Gx,
				real_t *Gy,
				real_t *Gxy)
{
}

} //!- namespace TW