#include "TW_paDIC_ICGN2D.h"

namespace TW{
namespace paDIC{

ICGN2D::ICGN2D(const cv::Mat& refImg,
			   const cv::Mat& tarImg,
			   int iStartX, int iStartY,
		       int iROIWidth, int iROIHeight,
		       int iSubsetX, int iSubsetY,
			   int iNumberX, int iNumberY,
		       int iNumIterations,
		       real_t fDeltaP)
			   : m_refImg(refImg)
			   , m_tarImg(tarImg)
			   , m_iStartX(iStartX)
			   , m_iStartY(iStartY)
			   , m_iROIWidth(iROIWidth)
			   , m_iROIHeight(iROIHeight)
			   , m_iSubsetX(iSubsetX)
			   , m_iSubsetY(iSubsetY)
			   , m_iNumberX(iNumberX)
			   , m_iNumberY(iNumberY)
			   , m_iNumIterations(iNumIterations)
			   , m_fDeltaP(fDeltaP)
{
	m_iPOINumber = m_iNumberX * m_iNumberY;
	m_iSubsetH = m_iSubsetY * 2 +1;
	m_iSubsetW = m_iSubsetX * 2 +1;
	m_iSubsetSize = m_iSubsetH * m_iSubsetW;
}

ICGN2D::~ICGN2D()
{}

} //!- namespace paDIC
} //!- namespace TW