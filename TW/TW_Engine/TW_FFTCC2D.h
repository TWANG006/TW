#ifndef TW_FFT_CC_H
#define TW_FFT_CC_H


#include "TW.h"
#include <vector>
#include <memory>

namespace TW{

	class TW_LIB_DLL_EXPORTS Fftcc2D
	{
	public:
		Fftcc2D(
			const int_t iROIWidth, const int_t iROIHeight,
			const int_t iSubsetX = 16, const int_t iSubsetY = 16,
			const int_t iGridSpaceX = 5, const int_t iGridSpaceY = 5,
			const int_t iMarginX = 3, const int_t iMarginY = 3);

		virtual ~Fftcc2D(){}

		//!- moveable but non-copyable class
		Fftcc2D() = delete;
		Fftcc2D(const Fftcc2D&) = delete;
		Fftcc2D& Fftcc2D::operator=(const Fftcc2D&) = delete;

		//!- TODO: make this class a callable object
		// void operator()(ref, tar, iU, iV, ZNCC)

		//!- Pure virtual functions
		//virtual void initializeFFTCC() = 0;
		//virtual void destroyFFTCC() = 0;

		//!- Inlined getters & setters
		inline int_t getNumPOIsX() const { return m_iNumPOIX; }
		inline int_t getNumPOIsY() const { return m_iNumPOIY; }
		inline int_t getNumPOIs() const { return (m_iNumPOIX*m_iNumPOIY); }
		void Fftcc2D::setROI(const int_t& iROIWidth, const int_t& iROIHeight);
		void Fftcc2D::setSubset(const int_t& iSubsetX, const int_t& iSubsetY);
		void Fftcc2D::setGridSpace(const int_t& iGridSpaceX, const int_t& iGridSpaceY);
		void Fftcc2D::setMargin(const int_t& iMarginX, const int_t& iMarginY);

	protected:
		bool recomputeNumPOI();

	protected:
		int_t m_iROIWidth, m_iROIHeight;
		int_t m_iSubsetX, m_iSubsetY;			//!- subsetSize = (2*m_iSubsetX+1)*(2*m_iSubsetY+1)
		int_t m_iGridSpaceX, m_iGridSpaceY;		//!- Number of pixels between each two POIs
		int_t m_iMarginX, m_iMarginY;			//!- Extra safe margin set for the ROI
		int_t m_iNumPOIX, m_iNumPOIY;			//!- Number of POIs = m_iNumPOIX*m_iNumPOIY
	};

	using ptrFFTCC2D = std::unique_ptr<Fftcc2D>;	//!- Smart pointer for FFTCC: only one Fftcc pointer 

} //!- namespace TW



#endif // ! TW_FFT_CC_H
