#include "TW_paDIC.h"

namespace TW{
namespace paDIC{

	//TW_LIB_DLL_EXPORTS GPUHandle_FFTCC g_cuHandle;

GPUHandle_FFTCC::GPUHandle_FFTCC()
	:m_d_fRefImg(nullptr)
	, m_d_fTarImg(nullptr)
	, m_d_iPOIXY(nullptr)
	, m_d_fV(nullptr)
	, m_d_fU(nullptr)
	, m_d_fZNCC(nullptr)
	, m_d_fSubset1(nullptr)
	, m_d_fSubset2(nullptr)
	, m_d_fSubsetC(nullptr)
	, m_d_fMod1(nullptr)
	, m_d_fMod2(nullptr)
	, m_dev_FreqDom1(nullptr)
	, m_dev_FreqDom2(nullptr)
	, m_dev_FreqDomfg(nullptr)
{}

GPUHandle_ICGN::GPUHandle_ICGN()
	: m_d_fRefImg(nullptr)
	, m_d_fTarImg(nullptr)
	, m_d_iPOIXY(nullptr)
	, m_d_fV(nullptr)
	, m_d_fU(nullptr)
	, m_d_fRx(nullptr)
	, m_d_fRy(nullptr)
	, m_d_fTx(nullptr)
	, m_d_fTy(nullptr)
	, m_d_fTxy(nullptr)
	, m_d_f4InterpolationLUT(nullptr)
	, m_d_iIterationNums(nullptr)
	, m_d_fSubsetR(nullptr)
	, m_d_fSubsetAveR(nullptr)
	, m_d_fSubsetT(nullptr)
	, m_d_fSubsetAveT(nullptr)
	, m_d_invHessian(nullptr)
	, m_d_RDescent(nullptr)
{}


}// namespace TW
}// namespace paDIC