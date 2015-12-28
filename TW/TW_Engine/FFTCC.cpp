#include "FFTCC.h"

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