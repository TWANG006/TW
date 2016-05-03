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
	int_t iMarginX = iImgWidth - (iStartX + iROIWidth) + 1;
	int_t iMarginY = iImgHeight -(iStartY + iROIHeight) + 1;

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
			if(Gx != nullptr && Gy != nullptr && Gxy!= nullptr)
			{
				for(int i=iStartY; i<(iStartY + iROIHeight); i++)
				{
					for(int j=iStartX; j<(iStartX + iROIWidth); j++)
					{
						int index = (i - iStartY)*iROIWidth + (j - iStartX);
						Gx[index] = 0.5 * real_t(image.at<uchar>(i,j+1) - image.at<uchar>(i,j-1));
						Gy[index] = 0.5 * real_t(image.at<uchar>(i+1,j) - image.at<uchar>(i-1,j));
						Gxy[index]= 0.25*real_t(image.at<uchar>(i+1,j+1) - image.at<uchar>(i-1,j+1)
							- image.at<uchar>(i+1,j-1) + image.at<uchar>(i-1,j-1));
					}
				}
			}
		}
		break;
	case TW::Quartic:
		if(iStartX<2 || iStartY<2 || iMarginX<2 || iMarginY <2)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			// TODO
		}
		break;
	case TW::Octic:
		if(iStartX<4 || iStartY<4 || iMarginX<4 || iMarginY <4)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			// TODO
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
	// TODO
}

void BicubicSplineCoefficients_s(//Inputs
	    						 const cv::Mat& image,
								 int_t iStartX, int_t iStartY,
								 int_t iROIWidth, int_t iROIHeight,
								 int_t iImgWidth, int_t iImgHeight,
								 //Output
								 real_t ****fBSpline)
{
	if( (iImgHeight - (iROIHeight + iStartY +1) < 0) || 
	    (iImgWidth  - (iROIWidth  + iStartX +1) < 0) )
	{
		throw("Error! Maximum boundary condition exceeded!");
	}

	real_t BSplineCP[4][4] = {
		 {  71 / 56.0, -19 / 56.0,   5 / 56.0,  -1 / 56.0 }, 
		 { -19 / 56.0,  95 / 56.0, -25 / 56.0,   5 / 56.0 }, 
		 {   5 / 56.0, -25 / 56.0,  95 / 56.0, -19 / 56.0 },
		 {  -1 / 56.0,   5 / 56.0, -19 / 56.0,  71 / 56.0 } 
	};
	real_t BSplineBase[4][4] = {
		{ -1 / 6.0,  3 / 6.0,  -3 / 6.0, 1 / 6.0 }, 
		{  3 / 6.0, -6 / 6.0,   3 / 6.0,       0 }, 
		{ -3 / 6.0,        0,   3 / 6.0,       0 }, 
		{  1 / 6.0,  4 / 6.0,   1 / 6.0,       0 } 
	};

	real_t fOmega[4][4];
	real_t fBeta[4][4];

	for(int i=0; i<iROIHeight; i++)
	{
		for(int j=0; j<iROIWidth; j++)
		{
			for(int k=0; k<4; k++)
			{
				for(int l=0; l<4; l++)
				{
					fOmega[k][l] = real_t(image.at<uchar>(i + iStartY - 1 + k, j + iStartX - 1 + l));
				}
			}
			for(int k=0; k<4; k++)
			{
				for(int l=0; l<4; l++)
				{
					fBeta[k][l] = 0;
					for(int m=0; m<4; m++)
					{
						for(int n=0; n<4; n++)
						{
							fBeta[k][l] += BSplineCP[k][m] * BSplineCP[l][n] * fOmega[n][m];
						}
					}
				}
			}
			for(int k=0; k<4; k++)
			{
				for(int l=0; l<4; l++)
				{
					fBSpline[i][j][k][l] = 0;
					for(int m=0; m<4; m++)
					{
						for(int n=0; n<4; n++)
						{
							fBSpline[i][j][k][l] += BSplineBase[k][m] * BSplineBase[l][n] * fBeta[n][m];
						}
					}
				}
			}
			for(int k=0; k<2; k++)
			{
				for(int l=0; l<4; l++)
				{
					float fTemp = fBSpline[i][j][k][l];
					fBSpline[i][j][k][l] = fBSpline[i][j][3 - k][3 - l];
					fBSpline[i][j][3 - k][3 - l] = fTemp; 
				}
			}
		}
	}
}

} //!- namespace TW