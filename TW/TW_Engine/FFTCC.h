#ifndef FFT_CC_H
#define FFT_CC_H

#include <vector>
#include <memory>

namespace TW{
	namespace Algorithm{

		class __declspec(dllexport) Fftcc
		{
		public:
			Fftcc(
				const int iROIWidth,
				const int iROIHeight,
				const int iSubsetX = 16,
				const int iSubsetY = 16,
				const int iGridSpaceX = 5,
				const int iGridSpaceY = 5,
				const int iMarginX = 3,
				const int iMarginY = 3);

			virtual ~Fftcc(){}

			//!- non-copyable class
			Fftcc() = delete;
			Fftcc(const Fftcc&) = delete;
			Fftcc& Fftcc::operator=(const Fftcc&) = delete;

			//!- Pure virtual functions
			virtual void initializeFFTCC() = 0;
			virtual void destroyFFTCC() = 0;

			//!- Inlined getters
			inline int getNumPOIsX() const { return m_iNumPOIX; }
			inline int getNumPOIsY() const { return m_iNumPOIY; }
			inline int getNumPOIs() const { return (m_iNumPOIX*m_iNumPOIY); }

		protected:
			int m_iROIWidth, m_iROIHeight;
			int m_iSubsetX, m_iSubsetY;			//!- subsetSize = (2*m_iSubsetX+1)*(2*m_iSubsetY+1)
			int m_iGridSpaceX, m_iGridSpaceY;	//!- Number of pixels between each two POIs
			int m_iMarginX, m_iMarginY;			//!- Extra safe margin set for the ROI
			int m_iNumPOIX, m_iNumPOIY;			//!- Number of POIs = m_iNumPOIX*m_iNumPOIY
		};
	} //!- namespace TW
} //!- namespace FFTCC



#endif // ! FFT_CC_H
