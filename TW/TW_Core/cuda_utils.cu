#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust\extrema.h>
#include <thrust\device_ptr.h>

//---------------------------------------------------------------------------------------!
//---------------------------------------------------------------------------------------!
//-------------------------------------CUDA Kernels--------------------------------------!

__global__ void ConstructTexImage()
{

}

//------------------------------------/CUDA Kernels--------------------------------------!
//---------------------------------------------------------------------------------------!
//---------------------------------------------------------------------------------------!

//---------------------------------------------------------------------------------------!
//---------------------------------------------------------------------------------------!
//---------------------------------------Wrappers----------------------------------------!
void minMaxRWrapper(int *iU, int *iV, int iNU, int iNV,
				    int* iminU, int* imaxU,
					int* iminV, int* imaxV)
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

//---------------------------------------------------------------------------------------!
//---------------------------------------------------------------------------------------!
//--------------------------------------/Wrappers----------------------------------------!