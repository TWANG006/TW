#ifndef TW_CUFFTCC_2D_H
#define TW_CUFFTCC_2D_H

#include "TW.h"
#include "TW_paDIC_FFTCC2D.h"

#include <cuda_runtime.h>
#include <cufft.h>


namespace TW{
	namespace paDIC{

		/// \struct GPUHandle
		/// \brief This handle is for the computation of FFTCC of paDIC on GPU using CUDA
		/// This struct holds all the needed parameters of the parallel FFTCC algorithm.
		struct GPUHandle
		{

			uchar *m_d_fRefImg;			// Reference image
			uchar *m_d_fTarImg;			// Target image

			int_t *m_d_iPOIXY;			// POI positions on device
			int_t *m_d_iU;
			int_t *m_d_iV;
			real_t *m_d_fZNCC;

			real_t *m_d_fSubset1;
			real_t *m_d_fSubset2;
			real_t *m_d_fSubsetC;
			real_t *m_d_fMod1;
			real_t *m_d_fMod2;

			cufftHandle m_forwardPlanXY;
			cufftHandle m_reversePlanXY;
			cudafftComplex *m_dev_FreqDom1;
			cudafftComplex *m_dev_FreqDom2;
			cudafftComplex *m_dev_FreqDomfg;

			GPUHandle()
				:m_d_fRefImg(nullptr), m_d_fTarImg(nullptr), m_d_iPOIXY(nullptr),
				m_d_iV(nullptr), m_d_iU(nullptr), m_d_fZNCC(nullptr),
				m_d_fSubset1(nullptr), m_d_fSubset2(nullptr), m_d_fSubsetC(nullptr),
				m_d_fMod1(nullptr), m_d_fMod2(nullptr),
				m_dev_FreqDom1(nullptr), m_dev_FreqDom2(nullptr), m_dev_FreqDomfg(nullptr)
			{}
		};

		/// \class cuFFTCC2D
		/// \brief The class for the parallel Ffftcc2D algorithm performed under the 
		/// paDIC method on GPU using CUDA
		class TW_LIB_DLL_EXPORTS cuFFTCC2D : public Fftcc2D
		{
		public:
						
			cuFFTCC2D(
				const int_t iROIWidth, const int_t iROIHeight,
				const int_t iSubsetX = 16, const int_t iSubsetY = 16,
				const int_t iGridSpaceX = 5, const int_t iGridSpaceY = 5,
				const int_t iMarginX = 3, const int_t iMarginY = 3);

			~cuFFTCC2D();

			// !-----------------------------High level method----------------------
			
			/// \brief Initialize the FFTCC algorithm, including allocating both device and host 
			/// memory space. NOTE: no need to pre-allocate memory for iU, iV and fZNCC, but the 
			/// de-allocation should be done manully. Allocate memory for refImg and tarImg, copy
			/// refImg to GPU memory space, make the CUFFT plan, allocate GPU memory space for CUFFT.
			///
			/// \param iU displacement field of all POIs in x direction on host
			/// \param iV displacement field of all POIs in y direction on host
			/// \param fZNCC ZNCC coefficients of all POIs 
			/// \param refImg input reference image
			virtual void InitializeFFTCC(
				// Output
				int_t**& iU,
				int_t**& iV,
				real_t**& fZNCC,
				// Input
				const cv::Mat& refImg) override;
			
			/// \brief
			virtual void ComputeFFTCC(
				// Output
				int_t**& iU,
				int_t**& iV,
				real_t**& fZNCC,
				// Input
				const cv::Mat& tarImg) override;

			// --------------------------High level method end-------------------------!

			


			// !-------------------------Low level GPU methods-----------------------
			/// \brief Initialize the FFTCC algorithm, but the result is not copied back
			/// to host memory. NOTE: i_d_U, i_d_V and f_d_ZNCC needs not to be pre-allocated.
			/// 
			/// \param...
			virtual void cuInitializeFFTCC(
				// Output
				int_t**& i_d_U,
				int_t**& i_d_V,
				real_t**& f_d_ZNCC,
				// Input
				const cv::Mat& refImg);
			virtual void cuComputeFFTCC(
				// Output
				int_t**& i_d_U,
				int_t**& i_d_V,
				real_t**& f_d_ZNCC,
				// Input
				const cv::Mat& refImg);
			// --------------------------Low level method end-------------------------!

			virtual void DestroyFFTCC() override;
			void resetRefImg(const cv::Mat& refImg);

		private:
			GPUHandle m_cuHandle;		// GPU structures to hold required data
			bool isLowLevelApiCalled;	// check if the low level apis in use
			bool isDestroyed;
			//cv::cuda::HostMem m_plRef;	// Page-locked host memory for ref image
			//cv::cuda::HostMem m_plTar;	// Page-locked host memory for tar image
		};
	}
}

#endif // !TW_CUFFTCC_2D_H
