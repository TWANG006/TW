#include "TW_paDIC_cuFFTCC2D.h"
#include <stdexcept>

namespace TW{
	namespace paDIC{

		cuFFTCC2D::cuFFTCC2D(
			const int iROIWidth,
			const int iROIHeight,
			const int iSubsetX,
			const int iSubsetY,
			const int iGridSpaceX,
			const int iGridSpaceY,
			const int iMarginX,
			const int iMarginY)
			:Fftcc2D(
			iROIWidth,
			iROIHeight,
			iSubsetX,
			iSubsetY,
			iGridSpaceX,
			iGridSpaceY,
			iMarginX,
			iMarginY)
		{
			if (!recomputeNumPOI())
				throw std::logic_error("Number of POIs is below 0!");
		}

		void cuFFTCC2D::initializeFFTCC()
		{}

		void cuFFTCC2D::computeFFTCC()
		{}

		void cuFFTCC2D::destroyFFTCC()
		{}
	}
}