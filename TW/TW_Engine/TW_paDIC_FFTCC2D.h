#ifndef TW_FFT_CC_H
#define TW_FFT_CC_H


#include "TW.h"
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

			/// \brief Constructor that takes configuration parameters
			/// \param iROIWidth width of the ROI
			/// \param iROIHeight height of the ROI
			/// \param iSubsetX half size of the square subset in x direction
			/// \param iSubsetY half size of the square subset in y direction
			/// \param iGridSpaceX number of pixels between two POIs in x direction
			/// \param iGirdSpaceY number of pixels between two POIs in y direction
			/// \param iMarginX number of extra safe pixels at ROI boundary in x direction
			/// \param iMarginY number of extra safe pixels at ROI boundary in y direction
			Fftcc2D(// TODO: Add ROI start index
					const int_t iROIWidth, const int_t iROIHeight,
					const int_t iSubsetX = 16, const int_t iSubsetY = 16,
					const int_t iGridSpaceX = 5, const int_t iGridSpaceY = 5,
					const int_t iMarginX = 3, const int_t iMarginY = 3);

			virtual ~Fftcc2D(){}

			//!- moveable but non-copyable class
			Fftcc2D() = delete;
			Fftcc2D(const Fftcc2D&) = delete;
			Fftcc2D& Fftcc2D::operator=(const Fftcc2D&) = delete;

			//!- make this class a callable object
			//void operator()(
			//	const std::vector<real_t>& refImg,	// reference Image vector 
			//	const std::vector<real_t>& tarImg,	// target Image vector 
			//	iU, iV, ZNCC) = 0;

			//!- Pure virtual functions
			virtual void InitializeFFTCC(// Output
										 int_t**& iU,
										 int_t**& iV,
										 real_t**& fZNCC,
										 // Input
										 const cv::Mat& refImg) = 0;
			virtual void ComputeFFTCC(// Output
									  int_t**& iU,
									  int_t**& iV,
									  real_t**& fZNCC,
									  // Input
									  const cv::Mat& tarImg) = 0;
			virtual void DestroyFFTCC(int_t**& iU,
					  				  int_t**& iV,
									  real_t**& fZNCC) = 0;
			
			//!- Virtual Functions
			virtual void cuInitializeFFTCC(// Output
						  				   int_t*& i_d_U,
										   int_t*& i_d_V,
										   real_t*& f_d_ZNCC,
										   // Input
										   const cv::Mat& refImg);
			virtual void cuComputeFFTCC(// Output
					   				    int_t*& i_d_U,
										int_t*& i_d_V,
										real_t*& f_d_ZNCC,
										// Input
										const cv::Mat& tarImg);
			virtual void cuDestroyFFTCC(int_t *& i_d_U,
										int_t *& i_d_V,
										real_t*& f_d_ZNCC);


			//!- Inlined getters & setters
			inline int_t GetNumPOIsX() const { return m_iNumPOIX; }
			inline int_t GetNumPOIsY() const { return m_iNumPOIY; }
			inline int_t GetNumPOIs() const { return (m_iNumPOIX*m_iNumPOIY); }
			inline int_t GetROISize() const { return (m_iROIWidth* m_iROIHeight); }
			/*void Fftcc2D::setROI(const int_t& iROIWidth, const int_t& iROIHeight);
			void Fftcc2D::setSubset(const int_t& iSubsetX, const int_t& iSubsetY);
			void Fftcc2D::setGridSpace(const int_t& iGridSpaceX, const int_t& iGridSpaceY);
			void Fftcc2D::setMargin(const int_t& iMarginX, const int_t& iMarginY);*/

		protected:
			bool recomputeNumPOI();
		
		protected:
			int_t m_iImgWidth, m_iImgHeight;
			int_t m_iROIWidth, m_iROIHeight;
			int_t m_iSubsetX, m_iSubsetY;			//!- subsetSize = (2*m_iSubsetX+1)*(2*m_iSubsetY+1)
			int_t m_iGridSpaceX, m_iGridSpaceY;		//!- Number of pixels between each two POIs
			int_t m_iMarginX, m_iMarginY;			//!- Extra safe margin set for the ROI
			int_t m_iNumPOIX, m_iNumPOIY;			//!- Number of POIs = m_iNumPOIX*m_iNumPOIY
		};

		using ptrFFTCC2D = std::unique_ptr<Fftcc2D>;	//!- Smart pointer for FFTCC: only one Fftcc pointer 
	} //!- namespace paDIC
} //!- namespace TW



#endif // ! TW_FFT_CC_H
