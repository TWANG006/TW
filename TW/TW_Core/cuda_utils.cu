#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust\extrema.h>
#include <thrust\device_ptr.h>

//---------------------------------------------------------------------------------------!
//---------------------------------------------------------------------------------------!
//-------------------------------------CUDA Kernels--------------------------------------!

// 1 Kb color map: from blue to red
 __constant__ unsigned int texture_data[256] =
    {
		0xFF830000, 0xFF870000, 0xFF8B0000, 0xFF8F0000, 0xFF930000, 0xFF970000, 0xFF9B0000, 0xFF9F0000,
		0xFFA30000, 0xFFA70000, 0xFFAB0000, 0xFFAF0000, 0xFFB30000, 0xFFB70000, 0xFFBB0000, 0xFFBF0000,
		0xFFC30000, 0xFFC70000, 0xFFCB0000, 0xFFCF0000, 0xFFD30000, 0xFFD70000, 0xFFDB0000, 0xFFDF0000,
		0xFFE30000, 0xFFE70000, 0xFFEB0000, 0xFFEF0000, 0xFFF30000, 0xFFF70000, 0xFFFB0000, 0xFFFF0000,
		0xFFFF0400, 0xFFFF0800, 0xFFFF0C00, 0xFFFF1000, 0xFFFF1400, 0xFFFF1800, 0xFFFF1C00, 0xFFFF2000,
		0xFFFF2400, 0xFFFF2800, 0xFFFF2C00, 0xFFFF3000, 0xFFFF3400, 0xFFFF3800, 0xFFFF3C00, 0xFFFF4000,
		0xFFFF4400, 0xFFFF4800, 0xFFFF4C00, 0xFFFF5000, 0xFFFF5400, 0xFFFF5800, 0xFFFF5C00, 0xFFFF6000,
		0xFFFF6400, 0xFFFF6800, 0xFFFF6C00, 0xFFFF7000, 0xFFFF7400, 0xFFFF7800, 0xFFFF7C00, 0xFFFF8000,
		0xFFFF8300, 0xFFFF8700, 0xFFFF8B00, 0xFFFF8F00, 0xFFFF9300, 0xFFFF9700, 0xFFFF9B00, 0xFFFF9F00,
		0xFFFFA300, 0xFFFFA700, 0xFFFFAB00, 0xFFFFAF00, 0xFFFFB300, 0xFFFFB700, 0xFFFFBB00, 0xFFFFBF00,
		0xFFFFC300, 0xFFFFC700, 0xFFFFCB00, 0xFFFFCF00, 0xFFFFD300, 0xFFFFD700, 0xFFFFDB00, 0xFFFFDF00,
		0xFFFFE300, 0xFFFFE700, 0xFFFFEB00, 0xFFFFEF00, 0xFFFFF300, 0xFFFFF700, 0xFFFFFB00, 0xFFFFFF00,
		0xFFFBFF04, 0xFFF7FF08, 0xFFF3FF0C, 0xFFEFFF10, 0xFFEBFF14, 0xFFE7FF18, 0xFFE3FF1C, 0xFFDFFF20,
		0xFFDBFF24, 0xFFD7FF28, 0xFFD3FF2C, 0xFFCFFF30, 0xFFCBFF34, 0xFFC7FF38, 0xFFC3FF3C, 0xFFBFFF40,
		0xFFBBFF44, 0xFFB7FF48, 0xFFB3FF4C, 0xFFAFFF50, 0xFFABFF54, 0xFFA7FF58, 0xFFA3FF5C, 0xFF9FFF60,
		0xFF9BFF64, 0xFF97FF68, 0xFF93FF6C, 0xFF8FFF70, 0xFF8BFF74, 0xFF87FF78, 0xFF83FF7C, 0xFF80FF80,
		0xFF7CFF83, 0xFF78FF87, 0xFF74FF8B, 0xFF70FF8F, 0xFF6CFF93, 0xFF68FF97, 0xFF64FF9B, 0xFF60FF9F,
		0xFF5CFFA3, 0xFF58FFA7, 0xFF54FFAB, 0xFF50FFAF, 0xFF4CFFB3, 0xFF48FFB7, 0xFF44FFBB, 0xFF40FFBF,
		0xFF3CFFC3, 0xFF38FFC7, 0xFF34FFCB, 0xFF30FFCF, 0xFF2CFFD3, 0xFF28FFD7, 0xFF24FFDB, 0xFF20FFDF,
		0xFF1CFFE3, 0xFF18FFE7, 0xFF14FFEB, 0xFF10FFEF, 0xFF0CFFF3, 0xFF08FFF7, 0xFF04FFFB, 0xFF00FFFF,
		0xFF00FBFF, 0xFF00F7FF, 0xFF00F3FF, 0xFF00EFFF, 0xFF00EBFF, 0xFF00E7FF, 0xFF00E3FF, 0xFF00DFFF,
		0xFF00DBFF, 0xFF00D7FF, 0xFF00D3FF, 0xFF00CFFF, 0xFF00CBFF, 0xFF00C7FF, 0xFF00C3FF, 0xFF00BFFF,
		0xFF00BBFF, 0xFF00B7FF, 0xFF00B3FF, 0xFF00AFFF, 0xFF00ABFF, 0xFF00A7FF, 0xFF00A3FF, 0xFF009FFF,
		0xFF009BFF, 0xFF0097FF, 0xFF0093FF, 0xFF008FFF, 0xFF008BFF, 0xFF0087FF, 0xFF0083FF, 0xFF0080FF,
		0xFF007CFF, 0xFF0078FF, 0xFF0074FF, 0xFF0070FF, 0xFF006CFF, 0xFF0068FF, 0xFF0064FF, 0xFF0060FF,
		0xFF005CFF, 0xFF0058FF, 0xFF0054FF, 0xFF0050FF, 0xFF004CFF, 0xFF0048FF, 0xFF0044FF, 0xFF0040FF,
		0xFF003CFF, 0xFF0038FF, 0xFF0034FF, 0xFF0030FF, 0xFF002CFF, 0xFF0028FF, 0xFF0024FF, 0xFF0020FF,
		0xFF001CFF, 0xFF0018FF, 0xFF0014FF, 0xFF0010FF, 0xFF000CFF, 0xFF0008FF, 0xFF0004FF, 0xFF0000FF,
		0xFF0000FB, 0xFF0000F7, 0xFF0000F3, 0xFF0000EF, 0xFF0000EB, 0xFF0000E7, 0xFF0000E3, 0xFF0000DF,
		0xFF0000DB, 0xFF0000D7, 0xFF0000D3, 0xFF0000CF, 0xFF0000CB, 0xFF0000C7, 0xFF0000C3, 0xFF0000BF,
		0xFF0000BB, 0xFF0000B7, 0xFF0000B3, 0xFF0000AF, 0xFF0000AB, 0xFF0000A7, 0xFF0000A3, 0xFF00009F,
		0xFF00009B, 0xFF000097, 0xFF000093, 0xFF00008F, 0xFF00008B, 0xFF000087, 0xFF000083, 0xFF000080
    };

__global__ void constructTextImage_Kernel(// Outputs
										  uint1* texImgU,
										  uint1* texImgV,
										  // Inputs
										  int* iU,
										  int* iV,
										  int iN,
										  int iMaxU, int iMinU,
										  int iMaxV, int iMinV)
{

}

//------------------------------------/CUDA Kernels--------------------------------------!
//---------------------------------------------------------------------------------------!
//---------------------------------------------------------------------------------------!

//---------------------------------------------------------------------------------------!
//---------------------------------------------------------------------------------------!
//---------------------------------------Wrappers----------------------------------------!
void minMaxRWrapper(int *&iU, int *&iV, int iNU, int iNV,
				    int* &iminU, int* &imaxU,
					int* &iminV, int* &imaxV)
{
	using iThDevPtr = thrust::device_ptr<int>;

	// Use thrust to find max and min simultaneously
	iThDevPtr d_Uptr(iU);
	thrust::pair<iThDevPtr, iThDevPtr> result_u = thrust::minmax_element(d_Uptr, d_Uptr+iNU);
	// Cast the thrust device pointer to raw device pointer
	iminU = thrust::raw_pointer_cast(result_u.first);
	imaxU = thrust::raw_pointer_cast(result_u.second);

	// Same for iV
	iThDevPtr d_Vptr(iV);
	thrust::pair<iThDevPtr, iThDevPtr> result_v = thrust::minmax_element(d_Vptr, d_Vptr+iNV);
	// Cast the thrust device pointer to raw device pointer
	iminV = thrust::raw_pointer_cast(result_u.first);
	imaxV = thrust::raw_pointer_cast(result_u.second);
}


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
				   int &iROIHeight)
{

}

void constructTextImage()
{

}

//---------------------------------------------------------------------------------------!
//---------------------------------------------------------------------------------------!
//--------------------------------------/Wrappers----------------------------------------!