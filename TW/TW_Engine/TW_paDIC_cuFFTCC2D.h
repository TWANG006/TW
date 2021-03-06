#ifndef TW_CUFFTCC_2D_H
#define TW_CUFFTCC_2D_H

#include "TW.h"
#include "TW_paDIC_FFTCC2D.h"

#include <cuda_runtime.h>
#include <cufft.h>


namespace TW{
namespace paDIC{


/// \class cuFFTCC2D
/// \brief The class for the parallel Ffftcc2D algorithm performed under the 
/// paDIC method on GPU using CUDA
class TW_LIB_DLL_EXPORTS cuFFTCC2D : public Fftcc2D
{
public:
	GPUHandle_FFTCC g_cuHandle;

	/// \brief cuFFTCC2D Constructor that takes configuration parameters of the ROI
	///
	/// \param iROIWidth width of the ROI
	/// \param iROIHeight height of the ROI
	/// \param iSubsetX half size of the square subset in x direction
	/// \param iSubsetY half size of the square subset in y direction
	/// \param iGridSpaceX number of pixels between two POIs in x direction
	/// \param iGirdSpaceY number of pixels between two POIs in y direction
	/// \param iMarginX number of extra safe pixels at ROI boundary in x direction
	/// \param iMarginY number of extra safe pixels at ROI boundary in y direction
	cuFFTCC2D(const int_t iROIWidth,      const int_t iROIHeight,
			  const int_t iSubsetX,		  const int_t iSubsetY,
			  const int_t iGridSpaceX,	  const int_t iGridSpace,
			  const int_t iMarginX,		  const int_t iMarginY);
	
	/// \brief cuFFTCC2D Constructor that takes configuration parameters of the whole image
	///
	/// \param iImgWidth width of the whole image
	/// \param iImgHeight height of the whole image
	/// \param iStartX x of the start point of the ROI 
	//  \param iStartY y of the start point of the ROI
	/// \param iROIWidth width of the ROI
	/// \param iROIHeight height of the ROI
	/// \param iSubsetX half size of the square subset in x direction
	/// \param iSubsetY half size of the square subset in y direction
	/// \param iGridSpaceX number of pixels between two POIs in x direction
	/// \param iGirdSpaceY number of pixels between two POIs in y direction
	/// \param iMarginX number of extra safe pixels at ROI boundary in x direction
	/// \param iMarginY number of extra safe pixels at ROI boundary in y direction
	cuFFTCC2D(const int_t iImgWidth,	   const int_t iImgHeight,
			  const int_t iROIWidth,       const int_t iROIHeight,
			  const int_t iStartX,         const int_t iStartY,
			  const int_t iSubsetX,        const int_t iSubsetY,
			  const int_t iGridSpaceX,     const int_t iGridSpaceY,
			  const int_t iMarginX,        const int_t iMarginY);

	~cuFFTCC2D();

	void setROI(const int_t& iStartX, const int_t& iStartY,
				const int_t& iROIWidth, const int_t& iROIHeight)
	{
		Fftcc2D::setROI(iStartX,iStartY,
						iROIWidth, iROIHeight);
	}

	// !-------------------------------------High level method----------------------------------

	/// \brief Initialize the FFTCC algorithm, including allocating both device and host 
	/// memory space. NOTE: no need to pre-allocate memory for iU, iV and fZNCC, but the 
	/// de-allocation should be done manully. Allocate memory for refImg and tarImg, copy
	/// refImg to GPU memory space, make the CUFFT plan, allocate GPU memory space for CUFFT.
	/// NOTE: cuInitializeFFTCC & InitializeFFTCC cannot be called at the same. 
	///
	/// \param iU displacement field of all POIs in x direction on host
	/// \param iV displacement field of all POIs in y direction on host
	/// \param fZNCC ZNCC coefficients of all POIs 
	/// \param refImg input reference image
	virtual void InitializeFFTCC(// Output
								 real_t**& fU,
								 real_t**& fV,
								 real_t**& fZNCC,
								 // Input
								 const cv::Mat& refImg);
			
	/// \brief Execute the FFTCC algorithm, including FFT and max-finding
	/// 
	/// \param iU displacement field of all POIs in x direction on host to be computed
	/// \param iV displacement field of all POIs in x direction on host to be computed
	/// \param fZNCC ZNCC coefficients of all POIs on host to be computed
	virtual void ComputeFFTCC(// Output
				              real_t**& fU,
							  real_t**& fV,
							  real_t**& fZNCC,
							  // Input
							  const cv::Mat& tarImg);

	/// \brief Finalize the FFTCC computation: deallocate memory on host & device, destroy
	/// the CUFFT plans
	///
	/// \param iU displacement field of all POIs in x direction on host
	/// \param iV displacement field of all POIs in x direction on host
	/// \param fZNCC ZNCC coefficients of all POIs on host
	virtual void DestroyFFTCC(real_t**& fU,
							  real_t**& fV,
							  real_t**& fZNCC);

	/// \brief Update the reference image 
	/// NOTE: This function should only be called after calling InitializeFFTCC 
	/// or cuInitializeFFTCC
	/// 
	/// \param refImg the new cv::Mat reference image
	virtual void ResetRefImg(const cv::Mat& refImg) override;
	virtual void SetTarImg(const cv::Mat& tarImg) override;

	// ---------------------------------High level method end-----------------------------------!


	// !---------------------------------Low level GPU methods-----------------------------------

	/// \brief Initialize the FFTCC algorithm on GPU. Allocate memory for i_d_U, i_d_V
	/// and f_d_ZNCC; copy the refImg to device memory.
	/// NOTE: isLowLevelApiCalled is set to be true when using this initialization function. 
	/// cuInitializeFFTCC & InitializeFFTCC cannot be called at the same. 
	/// 
	/// \param i_d_U displacement field of all POIs in x direction on device
	/// \param i_d_V displacement field of all POIs in y direction on device
	/// \param f_d_ZNCC ZNCC coefficients of all POIs on device
	/// \param refImg input reference image
	virtual void cuInitializeFFTCC(// Output
								   real_t *& f_d_U,
								   real_t *& f_d_V,
								   real_t*& f_d_ZNCC,
								   // Input
								   const cv::Mat& refImg);

	/// \brief Execute the FFTCC algorithm, including FFT and max-finding. The result is not passed 
	/// from device to host memory. The resources have been initialized by the cuInitializeFFTCC()
	/// 
	/// \param i_d_U displacement field of all POIs in x direction on host to be computed
	/// \param i_d_V displacement field of all POIs in x direction on host to be computed
	/// \param f_d_ZNCC ZNCC coefficients of all POIs on host to be computed
	virtual void cuComputeFFTCC(// Output
								real_t *& f_d_U,
								real_t *& f_d_V,
								real_t*& f_d_ZNCC,
								// Input
								const cv::Mat& tarImg);

	/// \brief Finalize the FFTCC computation: deallocate memory on device, destroy
	/// the CUFFT plans
	/// \param i_d_U displacement field of all POIs in x direction on device
	/// \param i_d_V displacement field of all POIs in y direction on device
	/// \param f_d_ZNCC ZNCC coefficients of all POIs on device
	/// \param refImg input reference image
	virtual void cuDestroyFFTCC(real_t *& f_d_U,
								real_t *& f_d_V,
								real_t*& f_d_ZNCC);

	// -----------------------------------Low level method end--------------------------------------!



private:
	bool isLowLevelApiCalled;		// check if the low level apis in use
	bool isDestroyed;				// check if the FFTCC is completed
	//cv::cuda::HostMem m_plRef;	// Page-locked host memory for ref image
	//cv::cuda::HostMem m_plTar;	// Page-locked host memory for tar image
};

} //!- namespace paDIC
} //!- namespace TW

#endif // !TW_CUFFTCC_2D_H
