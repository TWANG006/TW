#ifndef TW_PADIC_FFT_CC_H
#define TW_PADIC_FFT_CC_H


#include "TW_paDIC.h"
#include <vector>
#include <memory>

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\core\cuda.hpp>

namespace TW{
namespace paDIC{

/// \class Fftcc2D
/// \brief The base class for the Ffftcc2D algorithm performed under the 
/// paDIC method
class TW_LIB_DLL_EXPORTS Fftcc2D
{
public:

	/// \brief Constructor that takes configuration parameters of the whole image
	///
	/// \param iImgWidth width of the whole image
	/// \param iImgHeight height of the whole image
	/// \param iStartX x of the start point of the ROI 
	//  \param iStartY y of the start point of the ROI
	/// \param iROIWidth width of the ROI
	/// \param iROIHeight height of the ROI
	/// \param iSubsetX half size of the square subset in x direction
	/// \param iSubsetY half size of the square subset in y direction
	/// \param iGridSpaceX number of pixels between two POIs in x direction
	/// \param iGirdSpaceY number of pixels between two POIs in y direction
	/// \param iMarginX number of extra safe pixels at ROI boundary in x direction
	/// \param iMarginY number of extra safe pixels at ROI boundary in y direction
	Fftcc2D(const int_t iImgWidth, const int_t iImgHeight,
			const int_t iStartX, const int_t iStartY,
			const int_t iROIWidth, const int_t iROIHeight,
			const int_t iSubsetX, const int_t iSubsetY,
			const int_t iGridSpaceX, const int_t iGridSpaceY,
			const int_t iMarginX, const int_t iMarginY);

	/// \brief Constructor that takes configuration parameters of the ROI
	///
	/// \param iROIWidth width of the ROI
	/// \param iROIHeight height of the ROI
	/// \param iSubsetX half size of the square subset in x direction
	/// \param iSubsetY half size of the square subset in y direction
	/// \param iGridSpaceX number of pixels between two POIs in x direction
	/// \param iGirdSpaceY number of pixels between two POIs in y direction
	/// \param iMarginX number of extra safe pixels at ROI boundary in x direction
	/// \param iMarginY number of extra safe pixels at ROI boundary in y direction
	Fftcc2D(const int_t iROIWidth, const int_t iROIHeight,
			const int_t iSubsetX, const int_t iSubsetY,
			const int_t iGridSpaceX, const int_t iGridSpaceY,
			const int_t iMarginX, const int_t iMarginY);

	virtual ~Fftcc2D(){}

	//!- moveable but non-copyable class
	Fftcc2D() = delete;
	Fftcc2D(const Fftcc2D&) = delete;
	Fftcc2D& Fftcc2D::operator=(const Fftcc2D&) = delete;

	//!- make this class a callable object TODO
	//void operator()(
	//	const std::vector<real_t>& refImg,	// reference Image vector 
	//	const std::vector<real_t>& tarImg,	// target Image vector 
	//	iU, iV, ZNCC) = 0;

	//!- Pure virtual functions
	//virtual void InitializeFFTCC(// Output
	//							 real_t**& fU,
	//							 real_t**& fV,
	//							 real_t**& fZNCC,
	//							 // Input
	//							 const cv::Mat& refImg) = 0;

	//virtual void ComputeFFTCC(// Output
	//						  real_t**& fU,
	//						  real_t**& fV,
	//						  real_t**& fZNCC,
	//						  // Input
	//						  const cv::Mat& tarImg) = 0;

	//virtual void DestroyFFTCC(real_t**& fU,
	//		  				  real_t**& fV,
	//						  real_t**& fZNCC) = 0;

	/// \brief Reset the Reference image with refImg
	///
	/// \param refImg reference image to be used
	virtual void ResetRefImg(const cv::Mat& refImg) = 0;


	//!- Inlined getters & setters
	inline int_t GetNumPOIsX() const { return m_iNumPOIX; }
	inline int_t GetNumPOIsY() const { return m_iNumPOIY; }
	inline int_t GetNumPOIs() const { return (m_iNumPOIX*m_iNumPOIY); }
	inline int_t GetROISize() const { return (m_iROIWidth* m_iROIHeight); }
	inline int_t GetImgSize() const { return (m_iImgHeight == -1 || m_iImgWidth ==-1)? GetROISize(): m_iImgHeight*m_iImgWidth; }
	virtual void setROI(const int_t& iStartX,   const int_t& iStartY,
						const int_t& iROIWidth, const int_t& iROIHeight);
	/*void Fftcc2D::setSubset(const int_t& iSubsetX, const int_t& iSubsetY);
	void Fftcc2D::setGridSpace(const int_t& iGridSpaceX, const int_t& iGridSpaceY);
	void Fftcc2D::setMargin(const int_t& iMarginX, const int_t& iMarginY);*/

protected:
	bool recomputeNumPOI();
	
protected:
	bool  m_isWholeImgUsed;					//!- Whether the calculation is based on the whole image or ROI
	int_t m_iImgWidth, m_iImgHeight;		//!- Whole Image Size
	int_t m_iROIWidth, m_iROIHeight;		//!- ROIsize
	int_t m_iStartX, m_iStartY;				//!- ROI top-left point
	int_t m_iSubsetX, m_iSubsetY;			//!- subsetSize = (2*m_iSubsetX+1)*(2*m_iSubsetY+1)
	int_t m_iGridSpaceX, m_iGridSpaceY;		//!- Number of pixels between each two POIs
	int_t m_iMarginX, m_iMarginY;			//!- Extra safe margin set for the ROI
	int_t m_iNumPOIX, m_iNumPOIY;			//!- Number of POIs = m_iNumPOIX*m_iNumPOIY
};

using ptrFFTCC2D = std::unique_ptr<Fftcc2D>;	//!- Smart pointer for FFTCC: only one Fftcc pointer 

} //!- namespace paDIC
} //!- namespace TW



#endif // ! TW_PADIC_FFT_CC_H
