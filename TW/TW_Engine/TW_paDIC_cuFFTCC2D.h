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
			int_t *m_d_iPOIXY;			// POI positions on device
			int_t *m_d_iU;
			int_t *m_d_iV;
			real_t *m_d_fZNCC;

			real_t *m_d_fSubset1;
			real_t *m_d_fSubset2;
			real_t *m_d_fSubsetC;
			real_t *m_d_fMod1;
			real_t *m_d_fMod2;

			real_t *m_d_fRefImg;
			real_t *m_d_fTarImg;

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

			virtual void initializeFFTCC() override;
			virtual void computeFFTCC() override;
			virtual void destroyFFTCC() override;

			//!- Low level GPU methods
			void resetRefImg(std::vector<intensity_t> refImg);
			void cuComputeFFTCC();

		private:
			GPUHandle m_cuHandle;
		};
	}
}

#endif // !TW_CUFFTCC_2D_H
