#include "TW_utils.h"
#include <iostream>
#include <omp.h>
#include "TW_MemManager.h"

namespace TW
{
void ComputePOIPositions_s(// Output
					   	   int_t ***&Out_h_iPXY,			// Return the host handle
						   // Inputs
						   int_t iNumberX, int_t iNumberY,
						   int_t iMarginX, int_t iMarginY,
						   int_t iSubsetX, int_t iSubsetY,
						   int_t iGridSpaceX, int_t iGridSpaceY)
{
	// Allocate memory
	hcreateptr<int_t>(Out_h_iPXY, iNumberY, iNumberX, 2);

	for (int i = 0; i < iNumberY; i++)
	{
		for (int j = 0; j < iNumberX; j++)
		{
			Out_h_iPXY[i][j][0] = iMarginY + iSubsetY + i * iGridSpaceY;
			Out_h_iPXY[i][j][1] = iMarginX + iSubsetX + j * iGridSpaceX;
		}
	}
}

void ComputePOIPositions_s(// Output
					   	   int_t ***&Out_h_iPXY,			// Return the host handle
						   // Inputs
						   int_t iStartX, int_t iStartY,
						   int_t iNumberX, int_t iNumberY,
						   int_t iMarginX, int_t iMarginY,
						   int_t iSubsetX, int_t iSubsetY,
						   int_t iGridSpaceX, int_t iGridSpaceY)
{
	// Allocate memory
	hcreateptr<int_t>(Out_h_iPXY, iNumberY, iNumberX, 2);

	for (int i = 0; i < iNumberY; i++)
	{
		for (int j = 0; j < iNumberX; j++)
		{
			Out_h_iPXY[i][j][0] = iStartY + iMarginY + iSubsetY + i * iGridSpaceY;
			Out_h_iPXY[i][j][1] = iStartX + iMarginX + iSubsetX + j * iGridSpaceX;
		}
	}
}

void ComputePOIPositions_m(// Output
						   int_t ***&Out_h_iPXY,			// Return the host handle
						   // Inputs
						   int_t iNumberX, int_t iNumberY,
						   int_t iMarginX, int_t iMarginY,
						   int_t iSubsetX, int_t iSubsetY,
						   int_t iGridSpaceX, int_t iGridSpaceY)
{
	// Allocate memory
	hcreateptr<int_t>(Out_h_iPXY, iNumberY, iNumberX, 2);

#pragma omp parallel for
	for (int i = 0; i < iNumberY; i++)
	{
		for (int j = 0; j < iNumberX; j++)
		{
			Out_h_iPXY[i][j][0] = iMarginY + iSubsetY + i * iGridSpaceY;
			Out_h_iPXY[i][j][1] = iMarginX + iSubsetX + j * iGridSpaceX;
		}
	}
}

void ComputePOIPositions_m(// Output
						   int_t ***&Out_h_iPXY,			// Return the host handle
						   // Inputs
						   int_t iStartX, int_t iStartY,
						   int_t iNumberX, int_t iNumberY,
						   int_t iMarginX, int_t iMarginY,
						   int_t iSubsetX, int_t iSubsetY,
						   int_t iGridSpaceX, int_t iGridSpaceY)
{
	// Allocate memory
	hcreateptr<int_t>(Out_h_iPXY, iNumberY, iNumberX, 2);

#pragma omp parallel for
	for (int i = 0; i < iNumberY; i++)
	{
		for (int j = 0; j < iNumberX; j++)
		{
			Out_h_iPXY[i][j][0] = iStartY + iMarginY + iSubsetY + i * iGridSpaceY;
			Out_h_iPXY[i][j][1] = iStartX + iMarginX + iSubsetX + j * iGridSpaceX;
		}
	}
}

void Gradient_s(//Inputs
    			const cv::Mat& image,
				int_t iStartX, int_t iStartY,
				int_t iROIWidth, int_t iROIHeight,
				int_t iImgWidth, int_t iImgHeight,
				AccuracyOrder accuracyOrder,
				//Output
				real_t **Gx,
				real_t **Gy)
{
	int_t iMarginX = iImgWidth - (iStartX + iROIWidth) + 1;
	int_t iMarginY = iImgHeight -(iStartY + iROIHeight) + 1;

	switch (accuracyOrder)
	{
	case TW::AccuracyOrder::Quadratic:
	default:
	{
		if (iStartX < 1 || iStartY < 1 || iMarginX < 1 || iMarginY < 1)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			for (int i = iStartY; i < (iStartY + iROIHeight); i++)
			{
				for (int j = iStartX; j < (iStartX + iROIWidth); j++)
				{
					//int index = (i - iStartY)*iROIWidth + (j - iStartX);
					Gx[i - iStartY][j - iStartX] = 0.5 * real_t(image.at<uchar>(i, j + 1) - image.at<uchar>(i, j - 1));
					Gy[i - iStartY][j - iStartX] = 0.5 * real_t(image.at<uchar>(i + 1, j) - image.at<uchar>(i - 1, j));
					/*Gxy[index]= 0.25*real_t(image.at<uchar>(i+1,j+1) - image.at<uchar>(i-1,j+1)
						- image.at<uchar>(i+1,j-1) + image.at<uchar>(i-1,j-1));*/
				}
			}

		}
		break;
	}
	case TW::AccuracyOrder::Quartic:
	{
		if (iStartX < 2 || iStartY < 2 || iMarginX < 2 || iMarginY < 2)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			// TODO
		}
		break;
	}

	case TW::AccuracyOrder::Octic:
	{
		if (iStartX < 4 || iStartY < 4 || iMarginX < 4 || iMarginY < 4)
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
}

void GradientXY_s(//Inputs
			      const cv::Mat& image,
				  int_t iStartX, int_t iStartY,
				  int_t iROIWidth, int_t iROIHeight,
				  int_t iImgWidth, int_t iImgHeight,
				  AccuracyOrder accuracyOrder,
				  //Output
				  real_t **Gx,
				  real_t **Gy,
				  real_t **Gxy)
{
	int_t iMarginX = iImgWidth - (iStartX + iROIWidth) + 1;
	int_t iMarginY = iImgHeight -(iStartY + iROIHeight) + 1;

	switch (accuracyOrder)
	{
	case TW::AccuracyOrder::Quadratic:
	default:
	{
		if (iStartX < 1 || iStartY < 1 || iMarginX < 1 || iMarginY < 1)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			for (int i = iStartY; i < (iStartY + iROIHeight); i++)
			{
				for (int j = iStartX; j < (iStartX + iROIWidth); j++)
				{
					//int index = (i - iStartY)*iROIWidth + (j - iStartX);
					Gx[i - iStartY][j - iStartX] = 0.5 * real_t(image.at<uchar>(i, j + 1) - image.at<uchar>(i, j - 1));
					Gy[i - iStartY][j - iStartX] = 0.5 * real_t(image.at<uchar>(i + 1, j) - image.at<uchar>(i - 1, j));
					Gxy[i - iStartY][j - iStartX] = 0.25* real_t(image.at<uchar>(i + 1, j + 1) - image.at<uchar>(i - 1, j + 1)
						- image.at<uchar>(i + 1, j - 1) + image.at<uchar>(i - 1, j - 1));
				}
			}

		}
		break;
	}

	case TW::AccuracyOrder::Quartic:
	{
		if (iStartX < 2 || iStartY < 2 || iMarginX < 2 || iMarginY < 2)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			// TODO
		}
		break;
	}

	case TW::AccuracyOrder::Octic:
	{
		if (iStartX < 4 || iStartY < 4 || iMarginX < 4 || iMarginY < 4)
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
}

void GradientXY_2Images_s(//Inputs
						  const cv::Mat& image1,
						  const cv::Mat& image2,
						  int_t iStartX, int_t iStartY,
						  int_t iROIWidth, int_t iROIHeight,
						  int_t iImgWidth, int_t iImgHeight,
						  AccuracyOrder accuracyOrder,
						  //Outputs
						  real_t **Fx,
						  real_t **Fy,
						  real_t **Gx,
						  real_t **Gy,
						  real_t **Gxy)
{
	int_t iMarginX = iImgWidth - (iStartX + iROIWidth) + 1;
	int_t iMarginY = iImgHeight -(iStartY + iROIHeight) + 1;

	switch (accuracyOrder)
	{
	case TW::AccuracyOrder::Quadratic:
	default:
	{
		if (iStartX < 1 || iStartY < 1 || iMarginX < 1 || iMarginY < 1)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			for (int i = iStartY; i < (iStartY + iROIHeight); i++)
			{
				for (int j = iStartX; j < (iStartX + iROIWidth); j++)
				{
					//int index = (i - iStartY)*iROIWidth + (j - iStartX);
					Fx[i - iStartY][j - iStartX] = 0.5 * real_t(image1.at<uchar>(i, j + 1) - image1.at<uchar>(i, j - 1));
					Fy[i - iStartY][j - iStartX] = 0.5 * real_t(image1.at<uchar>(i + 1, j) - image1.at<uchar>(i - 1, j));

					Gx[i - iStartY][j - iStartX] = 0.5 * real_t(image2.at<uchar>(i, j + 1) - image2.at<uchar>(i, j - 1));
					Gy[i - iStartY][j - iStartX] = 0.5 * real_t(image2.at<uchar>(i + 1, j) - image2.at<uchar>(i - 1, j));
					Gxy[i - iStartY][j - iStartX] = 0.25* real_t(image2.at<uchar>(i + 1, j + 1) - image2.at<uchar>(i - 1, j + 1)
						- image2.at<uchar>(i + 1, j - 1) + image2.at<uchar>(i - 1, j - 1));
				}
			}

		}
		break;
	}

	case TW::AccuracyOrder::Quartic:
	{
		if (iStartX < 2 || iStartY < 2 || iMarginX < 2 || iMarginY < 2)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			// TODO
		}
		break;
	}

	case TW::AccuracyOrder::Octic:
	{
		if (iStartX < 4 || iStartY < 4 || iMarginX < 4 || iMarginY < 4)
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
}

void Gradient_m(//Inputs
				const cv::Mat& image,
				int_t iStartX, int_t iStartY,
				int_t iROIWidth, int_t iROIHeight,
				int_t iImgWidth, int_t iImgHeight,
				AccuracyOrder accuracyOrder,
				//Output
				real_t **Gx,
				real_t **Gy)
{
	int_t iMarginX = iImgWidth - (iStartX + iROIWidth) + 1;
	int_t iMarginY = iImgHeight -(iStartY + iROIHeight) + 1;

	switch (accuracyOrder)
	{
	case TW::AccuracyOrder::Quadratic:
	default:
	{
		if (iStartX < 1 || iStartY < 1 || iMarginX < 1 || iMarginY < 1)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
#pragma omp parallel for
			for (int i = iStartY; i < (iStartY + iROIHeight); i++)
			{
				for (int j = iStartX; j < (iStartX + iROIWidth); j++)
				{
					//int index = (i - iStartY)*iROIWidth + (j - iStartX);
					Gx[i - iStartY][j - iStartX] = 0.5 * real_t(image.at<uchar>(i, j + 1) - image.at<uchar>(i, j - 1));
					Gy[i - iStartY][j - iStartX] = 0.5 * real_t(image.at<uchar>(i + 1, j) - image.at<uchar>(i - 1, j));
					/*Gxy[index]= 0.25*real_t(image.at<uchar>(i+1,j+1) - image.at<uchar>(i-1,j+1)
						- image.at<uchar>(i+1,j-1) + image.at<uchar>(i-1,j-1));*/
				}
			}

		}
		break;
	}
	case TW::AccuracyOrder::Quartic:
	{
		if (iStartX < 2 || iStartY < 2 || iMarginX < 2 || iMarginY < 2)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			// TODO
		}
		break;
	}

	case TW::AccuracyOrder::Octic:
	{
		if (iStartX < 4 || iStartY < 4 || iMarginX < 4 || iMarginY < 4)
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
}

void GradientXY_m(//Inputs
				  const cv::Mat& image,
				  int_t iStartX, int_t iStartY,
				  int_t iROIWidth, int_t iROIHeight,
				  int_t iImgWidth, int_t iImgHeight,
				  AccuracyOrder accuracyOrder,
				  //Output
				  real_t **Gx,
				  real_t **Gy,
				  real_t **Gxy)
{
	int_t iMarginX = iImgWidth - (iStartX + iROIWidth) + 1;
	int_t iMarginY = iImgHeight -(iStartY + iROIHeight) + 1;

	switch (accuracyOrder)
	{
	case TW::AccuracyOrder::Quadratic:
	default:
	{
		if (iStartX < 1 || iStartY < 1 || iMarginX < 1 || iMarginY < 1)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
#pragma omp parallel for
			for (int i = iStartY; i < (iStartY + iROIHeight); i++)
			{
				for (int j = iStartX; j < (iStartX + iROIWidth); j++)
				{
					//int index = (i - iStartY)*iROIWidth + (j - iStartX);
					Gx[i - iStartY][j - iStartX] = 0.5 * real_t(image.at<uchar>(i, j + 1) - image.at<uchar>(i, j - 1));
					Gy[i - iStartY][j - iStartX] = 0.5 * real_t(image.at<uchar>(i + 1, j) - image.at<uchar>(i - 1, j));
					Gxy[i - iStartY][j - iStartX] = 0.25* real_t(image.at<uchar>(i + 1, j + 1) - image.at<uchar>(i - 1, j + 1)
						- image.at<uchar>(i + 1, j - 1) + image.at<uchar>(i - 1, j - 1));
				}
			}

		}
		break;
	}

	case TW::AccuracyOrder::Quartic:
	{
		if (iStartX < 2 || iStartY < 2 || iMarginX < 2 || iMarginY < 2)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			// TODO
		}
		break;
	}

	case TW::AccuracyOrder::Octic:
	{
		if (iStartX < 4 || iStartY < 4 || iMarginX < 4 || iMarginY < 4)
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
}

void GradientXY_2Images_m(//Inputs
						  const cv::Mat& image1,
						  const cv::Mat& image2,
						  int_t iStartX, int_t iStartY,
						  int_t iROIWidth, int_t iROIHeight,
						  int_t iImgWidth, int_t iImgHeight,
						  AccuracyOrder accuracyOrder,
						  //Outputs
						  real_t **Fx,
						  real_t **Fy,
						  real_t **Gx,
						  real_t **Gy,
						  real_t **Gxy)
{
	int_t iMarginX = iImgWidth - (iStartX + iROIWidth) + 1;
	int_t iMarginY = iImgHeight -(iStartY + iROIHeight) + 1;

	switch (accuracyOrder)
	{
	case TW::AccuracyOrder::Quadratic:
	default:
	{
		if (iStartX < 1 || iStartY < 1 || iMarginX < 1 || iMarginY < 1)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
#pragma omp parallel for
			for (int i = iStartY; i < (iStartY + iROIHeight); i++)
			{
				for (int j = iStartX; j < (iStartX + iROIWidth); j++)
				{
					//int index = (i - iStartY)*iROIWidth + (j - iStartX);
					Fx[i - iStartY][j - iStartX] = 0.5 * real_t(image1.at<uchar>(i, j + 1) - image1.at<uchar>(i, j - 1));
					Fy[i - iStartY][j - iStartX] = 0.5 * real_t(image1.at<uchar>(i + 1, j) - image1.at<uchar>(i - 1, j));

					Gx[i - iStartY][j - iStartX] = 0.5 * real_t(image2.at<uchar>(i, j + 1) - image2.at<uchar>(i, j - 1));
					Gy[i - iStartY][j - iStartX] = 0.5 * real_t(image2.at<uchar>(i + 1, j) - image2.at<uchar>(i - 1, j));
					Gxy[i - iStartY][j - iStartX] = 0.25* real_t(image2.at<uchar>(i + 1, j + 1) - image2.at<uchar>(i - 1, j + 1)
						- image2.at<uchar>(i + 1, j - 1) + image2.at<uchar>(i - 1, j - 1));
				}
			}

		}
		break;
	}

	case TW::AccuracyOrder::Quartic:
	{
		if (iStartX < 2 || iStartY < 2 || iMarginX < 2 || iMarginY < 2)
		{
			throw("Error: Not enough boundary pixels for gradients!");
		}
		else
		{
			// TODO
		}
		break;
	}

	case TW::AccuracyOrder::Octic:
	{
		if (iStartX < 4 || iStartY < 4 || iMarginX < 4 || iMarginY < 4)
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
					fOmega[k][l] = static_cast<real_t>(image.at<uchar>(i + iStartY - 1 + k, j + iStartX - 1 + l));
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
					real_t fTemp = fBSpline[i][j][k][l];
					fBSpline[i][j][k][l] = fBSpline[i][j][3 - k][3 - l];
					fBSpline[i][j][3 - k][3 - l] = fTemp; 
				}
			}
		}
	}
}

void BicubicCoefficients_s(// Inputs
 						   const cv::Mat& image,
						   real_t **Tx,
						   real_t **Ty,
						   real_t **Txy,
						   int_t iStartX, int_t iStartY,
						   int_t iROIWidth, int_t iROIHeight,
						   int_t iImgWidth, int_t iImgHeight,
						   // Outputs
						   real_t ****fBicubic)
{
	if( (iImgHeight - (iROIHeight + iStartY +1) < 0) || 
	    (iImgWidth  - (iROIWidth  + iStartX +1) < 0) )
	{
		throw("Error! Maximum boundary condition exceeded!");
	}

	// The coefficient matrix of the Bicubic interpolation
	real_t fBicubicMatrix[16][16] = { 
		{  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 }, 
		{  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 }, 
		{ -3,  3,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 }, 
		{  2, -2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 }, 
		{  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0 }, 
		{  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0 }, 
		{  0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -2, -1,  0,  0 }, 
		{  0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  1,  1,  0,  0 },
		{ -3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0,  0,  0,  0,  0 }, 
		{  0,  0,  0,  0, -3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0 }, 
		{  9, -9, -9,  9,  6,  3, -6, -3,  6, -6,  3, -3,  4,  2,  2,  1 },
		{ -6,  6,  6, -6, -3, -3,  3,  3, -4,  4, -2,  2, -2, -2, -1, -1 }, 
		{  2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0 }, 
		{  0,  0,  0,  0,  2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0 }, 
		{ -6,  6,  6, -6, -4, -2,  4,  2, -3,  3, -3,  3, -2, -1, -2, -1 },
		{  4, -4, -4,  4,  2,  2, -2, -2,  2, -2,  2, -2,  1,  1,  1,  1 }};

	real_t fTao[16];
	real_t fAlpha[16];

	for(int i = 0; i < iROIHeight - 1; i++)
	{
		for(int j = 0; j < iROIWidth - 1; j++)
		{
			fTao[0] = static_cast<real_t>(image.at<uchar>(i + iStartY,     j + iStartX));
			fTao[1] = static_cast<real_t>(image.at<uchar>(i + iStartY,     j + iStartX + 1));
			fTao[2] = static_cast<real_t>(image.at<uchar>(i + iStartY + 1, j + iStartX));
			fTao[3] = static_cast<real_t>(image.at<uchar>(i + iStartY + 1, j + iStartX + 1));
			fTao[4] = Tx[i][j];
			fTao[5] = Tx[i][j + 1];
			fTao[6] = Tx[i + 1][j];
			fTao[7] = Tx[i + 1][j + 1];
			fTao[8] = Ty[i][j];
			fTao[9] = Ty[i][j + 1];
			fTao[10]= Ty[i + 1][j];
			fTao[11]= Ty[i + 1][j + 1];
			fTao[12]= Txy[i][j];
			fTao[13]= Txy[i][j + 1];
			fTao[14]= Txy[i + 1][j];
			fTao[15]= Txy[i + 1][j + 1]; 

			for(int k = 0; k < 16; k++)
			{
				fAlpha[k] = 0;
				for(int l = 0; l < 16; l++)
				{
					fAlpha[k] += fBicubicMatrix[k][l] * fTao[l];
				}
			}

			fBicubic[i][j][0][0] = fAlpha[0];
			fBicubic[i][j][0][1] = fAlpha[1];
			fBicubic[i][j][0][2] = fAlpha[2];
			fBicubic[i][j][0][3] = fAlpha[3];
			fBicubic[i][j][1][0] = fAlpha[4];
			fBicubic[i][j][1][1] = fAlpha[5];
			fBicubic[i][j][1][2] = fAlpha[6];
			fBicubic[i][j][1][3] = fAlpha[7];
			fBicubic[i][j][2][0] = fAlpha[8];
			fBicubic[i][j][2][1] = fAlpha[9];
			fBicubic[i][j][2][2] = fAlpha[10];
			fBicubic[i][j][2][3] = fAlpha[11];
			fBicubic[i][j][3][0] = fAlpha[12];
			fBicubic[i][j][3][1] = fAlpha[13];
			fBicubic[i][j][3][2] = fAlpha[14];
			fBicubic[i][j][3][3] = fAlpha[15];
		}
	}

	// Padding the boundary ones with zeros
	for (int i = 0; i < 4; i++)
	{
		for(int j = 0; j < 4; j++)
		{
			fBicubic[iROIHeight - 1][iROIWidth - 1][i][j] = 0;
		}
	}
}

void BicubicSplineCoefficients_m(// Inputs
								 const cv::Mat& image,
								 int_t iStartX, int_t iStartY,
								 int_t iROIWidth, int_t iROIHeight,
								 int_t iImgWidth, int_t iImgHeight,
								 // Output
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

#pragma omp parallel
	{
	real_t fOmega[4][4];
	real_t fBeta[4][4];

#pragma omp parallel for
	for(int i=0; i<iROIHeight; i++)
	{
		for(int j=0; j<iROIWidth; j++)
		{
			for(int k=0; k<4; k++)
			{
				for(int l=0; l<4; l++)
				{
					fOmega[k][l] = static_cast<real_t>(image.at<uchar>(i + iStartY - 1 + k, j + iStartX - 1 + l));
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
					real_t fTemp = fBSpline[i][j][k][l];
					fBSpline[i][j][k][l] = fBSpline[i][j][3 - k][3 - l];
					fBSpline[i][j][3 - k][3 - l] = fTemp; 
				}
			}
		}
	}
	}
}

void BicubicCoefficients_m(// Inputs
						   const cv::Mat& image,
						   real_t **Tx,
						   real_t **Ty,
						   real_t **Txy,
						   int_t iStartX, int_t iStartY,
						   int_t iROIWidth, int_t iROIHeight,
						   int_t iImgWidth, int_t iImgHeight,
						   // Outputs
						   real_t ****fBicubic)
{
	if( (iImgHeight - (iROIHeight + iStartY +1) < 0) || 
	    (iImgWidth  - (iROIWidth  + iStartX +1) < 0) )
	{
		throw("Error! Maximum boundary condition exceeded!");
	}

	// The coefficient matrix of the Bicubic interpolation
	real_t fBicubicMatrix[16][16] = { 
		{  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 }, 
		{  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 }, 
		{ -3,  3,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 }, 
		{  2, -2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 }, 
		{  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0 }, 
		{  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0 }, 
		{  0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -2, -1,  0,  0 }, 
		{  0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  1,  1,  0,  0 },
		{ -3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0,  0,  0,  0,  0 }, 
		{  0,  0,  0,  0, -3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0 }, 
		{  9, -9, -9,  9,  6,  3, -6, -3,  6, -6,  3, -3,  4,  2,  2,  1 },
		{ -6,  6,  6, -6, -3, -3,  3,  3, -4,  4, -2,  2, -2, -2, -1, -1 }, 
		{  2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0 }, 
		{  0,  0,  0,  0,  2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0 }, 
		{ -6,  6,  6, -6, -4, -2,  4,  2, -3,  3, -3,  3, -2, -1, -2, -1 },
		{  4, -4, -4,  4,  2,  2, -2, -2,  2, -2,  2, -2,  1,  1,  1,  1 }};

#pragma omp parallel
	{
	real_t fTao[16];
	real_t fAlpha[16];

#pragma omp parallel for
	for(int i = 0; i < iROIHeight - 1; i++)
	{
		for(int j = 0; j < iROIWidth - 1; j++)
		{
			fTao[0] = static_cast<real_t>(image.at<uchar>(i + iStartY,     j + iStartX));
			fTao[1] = static_cast<real_t>(image.at<uchar>(i + iStartY,     j + iStartX + 1));
			fTao[2] = static_cast<real_t>(image.at<uchar>(i + iStartY + 1, j + iStartX));
			fTao[3] = static_cast<real_t>(image.at<uchar>(i + iStartY + 1, j + iStartX + 1));
			fTao[4] = Tx[i][j];
			fTao[5] = Tx[i][j + 1];
			fTao[6] = Tx[i + 1][j];
			fTao[7] = Tx[i + 1][j + 1];
			fTao[8] = Ty[i][j];
			fTao[9] = Ty[i][j + 1];
			fTao[10]= Ty[i + 1][j];
			fTao[11]= Ty[i + 1][j + 1];
			fTao[12]= Txy[i][j];
			fTao[13]= Txy[i][j + 1];
			fTao[14]= Txy[i + 1][j];
			fTao[15]= Txy[i + 1][j + 1]; 

			for(int k = 0; k < 16; k++)
			{
				fAlpha[k] = 0;
				for(int l = 0; l < 16; l++)
				{
					fAlpha[k] += fBicubicMatrix[k][l] * fTao[l];
				}
			}

			fBicubic[i][j][0][0] = fAlpha[0];
			fBicubic[i][j][0][1] = fAlpha[1];
			fBicubic[i][j][0][2] = fAlpha[2];
			fBicubic[i][j][0][3] = fAlpha[3];
			fBicubic[i][j][1][0] = fAlpha[4];
			fBicubic[i][j][1][1] = fAlpha[5];
			fBicubic[i][j][1][2] = fAlpha[6];
			fBicubic[i][j][1][3] = fAlpha[7];
			fBicubic[i][j][2][0] = fAlpha[8];
			fBicubic[i][j][2][1] = fAlpha[9];
			fBicubic[i][j][2][2] = fAlpha[10];
			fBicubic[i][j][2][3] = fAlpha[11];
			fBicubic[i][j][3][0] = fAlpha[12];
			fBicubic[i][j][3][1] = fAlpha[13];
			fBicubic[i][j][3][2] = fAlpha[14];
			fBicubic[i][j][3][3] = fAlpha[15];
		}
	}
	}
	// Padding the boundary ones with zeros
	for (int i = 0; i < 4; i++)
	{
		for(int j = 0; j < 4; j++)
		{
			fBicubic[iROIHeight - 1][iROIWidth - 1][i][j] = 0;
		}
	}
}

} //!- namespace TW