#include "TW_paDIC_ICGN2D.h"

namespace TW{
namespace paDIC{

ICGN2D::ICGN2D(/*const cv::Mat& refImg,
			   const cv::Mat& tarImg,*/
			   //const cv::Mat& refImg,
			   int_t iImgWidth, int_t iImgHeight,
			   int_t iStartX, int_t iStartY,
		       int_t iROIWidth, int_t iROIHeight,
		       int_t iSubsetX, int_t iSubsetY,
			   int_t iNumberX, int_t iNumberY,
		       int_t iNumIterations,
		       real_t fDeltaP)
			   : m_iImgWidth(iImgWidth)
			   , m_iImgHeight(iImgHeight)
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
			   , m_isRefImgUpdated(false)
			   //, m_refImg(refImg)
{
	// Precompute parameters
	m_iPOINumber = m_iNumberX * m_iNumberY;
	m_iSubsetH = m_iSubsetY * 2 +1;
	m_iSubsetW = m_iSubsetX * 2 +1;
	m_iSubsetSize = m_iSubsetH * m_iSubsetW;
}

ICGN2D::ICGN2D(const cv::Mat& refImg,
			   int_t iImgWidth, int_t iImgHeight,
			   int_t iStartX, int_t iStartY,
		       int_t iROIWidth, int_t iROIHeight,
		       int_t iSubsetX, int_t iSubsetY,
			   int_t iNumberX, int_t iNumberY,
		       int_t iNumIterations,
		       real_t fDeltaP)
			   : m_iImgWidth(iImgWidth)
			   , m_iImgHeight(iImgHeight)
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
			   , m_isRefImgUpdated(true)
			   , m_refImg(refImg)
{
	// Precompute parameters
	m_iPOINumber = m_iNumberX * m_iNumberY;
	m_iSubsetH = m_iSubsetY * 2 +1;
	m_iSubsetW = m_iSubsetX * 2 +1;
	m_iSubsetSize = m_iSubsetH * m_iSubsetW;
}

ICGN2D::~ICGN2D()
{}

void ICGN2D::setROI(const int_t& iStartX,   const int_t& iStartY,
					const int_t& iROIWidth, const int_t& iROIHeight)
{
	m_iStartX = iStartX;
	m_iStartY = iStartY;
	m_iROIWidth = iROIWidth;
	m_iROIHeight = iROIHeight;
}

} //!- namespace paDIC
} //!- namespace TW