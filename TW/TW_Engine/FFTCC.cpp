#include "FFTCC.h"

namespace TW{
	namespace Algorithm{

		Fftcc::Fftcc(
			const int iROIWidth,
			const int iROIHeight,
			const int iSubsetX,
			const int iSubsetY,
			const int iGridSpaceX,
			const int iGridSpaceY,
			const int iMarginX,
			const int iMarginY)
			: m_iROIWidth(iROIWidth)
			, m_iROIHeight(iROIHeight)
			, m_iSubsetX(iSubsetX)
			, m_iSubsetY(iSubsetY)
			, m_iGridSpaceX(iGridSpaceX)
			, m_iGridSpaceY(iGridSpaceY)
			, m_iMarginX(iMarginX)
			, m_iMarginY(iMarginY)
		{
			m_iNumPOIX = int(floor((iROIWidth - m_iSubsetX * 2 - m_iMarginX * 2) / float(m_iGridSpaceX))) + 1;
			m_iNumPOIY = int(floor((iROIHeight - m_iSubsetY * 2 - m_iMarginY * 2) / float(m_iGridSpaceY))) + 1;
		}

	} // namespace TW
} // namespace Algorithm

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