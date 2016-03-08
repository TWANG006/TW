#include "TW_paDIC_FFTCC2D.h"

namespace TW{
namespace paDIC{

Fftcc2D::Fftcc2D(const int_t iImgWidth, const int_t iImgHeight,
				 const int_t iStartX, const int_t iStartY,
				 const int_t iROIWidth, const int_t iROIHeight,
				 const int_t iSubsetX, const int_t iSubsetY,
				 const int_t iGridSpaceX, const int_t iGridSpaceY,
				 const int_t iMarginX, const int_t iMarginY)
	: m_iImgWidth(iImgWidth)
	, m_iImgHeight(iImgHeight)
	, m_iROIWidth(iROIWidth)
	, m_iROIHeight(iROIHeight)
	, m_iStartX(iStartX)
	, m_iStartY(iStartY)
	, m_iSubsetX(iSubsetX)
	, m_iSubsetY(iSubsetY)
	, m_iGridSpaceX(iGridSpaceX)
	, m_iGridSpaceY(iGridSpaceY)
	, m_iMarginX(iMarginX)
	, m_iMarginY(iMarginY)
	, m_isWholeImgUsed(true)
{}

Fftcc2D::Fftcc2D(const int_t iROIWidth, const int_t iROIHeight,
				 const int_t iSubsetX, const int_t iSubsetY,
				 const int_t iGridSpaceX, const int_t iGridSpaceY,
				 const int_t iMarginX, const int_t iMarginY)
	: m_iImgWidth(-1)
	, m_iImgHeight(-1)
	, m_iROIWidth(iROIWidth)
	, m_iROIHeight(iROIHeight)
	, m_iStartX(-1)
	, m_iStartY(-1)
	, m_iSubsetX(iSubsetX)
	, m_iSubsetY(iSubsetY)
	, m_iGridSpaceX(iGridSpaceX)
	, m_iGridSpaceY(iGridSpaceY)
	, m_iMarginX(iMarginX)
	, m_iMarginY(iMarginY)
	, m_isWholeImgUsed(false)
{}
/*void Fftcc2D::setROI(const int_t& iROIWidth, const int_t& iROIHeight)
		{
			m_iROIWidth = iROIWidth;
			m_iROIHeight = iROIHeight;

			if (!recomputeNumPOI())
				throw std::logic_error("Number of POIs is below 0!");
		}
		void Fftcc2D::setSubset(const int_t& iSubsetX, const int_t& iSubsetY)
		{
			m_iSubsetX = iSubsetX;
			m_iSubsetY = iSubsetY;

			if (!recomputeNumPOI())
				throw std::logic_error("Number of POIs is below 0!");
		}
		void Fftcc2D::setGridSpace(const int_t& iGridSpaceX, const int_t& iGridSpaceY)
		{
			m_iGridSpaceX = iGridSpaceX;
			m_iGridSpaceY = iGridSpaceY;

			if (!recomputeNumPOI())
				throw std::logic_error("Number of POIs is below 0!");
		}
		void Fftcc2D::setMargin(const int_t& iMarginX, const int_t& iMarginY)
		{
			m_iMarginX = iMarginX;
			m_iMarginY = iMarginY;

			if (!recomputeNumPOI())
				throw std::logic_error("Number of POIs is below 0!");
		}*/

bool Fftcc2D::recomputeNumPOI()
{
	m_iNumPOIX = int_t(floor((m_iROIWidth - m_iSubsetX * 2 - m_iMarginX * 2) / real_t(m_iGridSpaceX))) + 1;
	m_iNumPOIY = int_t(floor((m_iROIHeight - m_iSubsetY * 2 - m_iMarginY * 2) / real_t(m_iGridSpaceY))) + 1;

	return ((m_iNumPOIX > 0 && m_iNumPOIY > 0) ? true : false);
}

void Fftcc2D::cuInitializeFFTCC(// Output
								int_t *& i_d_U,
								int_t *& i_d_V,
								real_t*& f_d_ZNCC,
								// Input
								const cv::Mat& refImg)
{}

void Fftcc2D::cuComputeFFTCC(// Output
							 int_t *& i_d_U,
							 int_t *& i_d_V,
							 real_t*& f_d_ZNCC,
							 // Input
							 const cv::Mat& tarImg)
{}

void Fftcc2D::cuDestroyFFTCC(int_t *& i_d_U,
							 int_t *& i_d_V,
						     real_t*& f_d_ZNCC)
{}

} //!- namespace paDIC
} //!- namespace TW

//!- Factory method
//class __declspec(dllexport) Fftcc_Factory
//{
//public:
//	enum Fftcc_Type{
//		SingleThread,
//		MultiThread,
//		GPU
//	};
//
//	static std::unique_ptr<Fftcc> createFFTCC(
//		const std::vector<float>& vecRefImg,
//		const int iSubsetX = 16,
//		const int iSubsetY = 16,
//		const int iGridSpaceX = 5,
//		const int iGridSpaceY = 5,
//		const int iMarginX = 3,
//		const int iMarginY = 3
//		)
//	{
//		switch (Fftcc_Type)
//		{
//		case TW::Algrithm::Fftcc_Factory::SingleThread:
//			return std::make_unique<CPUFftcc>(
//				vecRefImg)
//				break;
//		case TW::Algrithm::Fftcc_Factory::MultiThread:
//			break;
//		case TW::Algrithm::Fftcc_Factory::GPU:
//			break;
//		default:
//			break;
//		}
//	}
//
//	Fftcc_Factory() = delete;
//	Fftcc_Factory(const Fftcc_Factory&) = delete;
//	Fftcc_Factory& Fftcc_Factory::operator=(const Fftcc_Factory&) = delete;
//
//};