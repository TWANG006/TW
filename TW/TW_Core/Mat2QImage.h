#ifndef _MAT_2_QIMAGE_
#define _MAT_2_QIMAGE_

#include <QImage>
#include <QPixmap>
#include <opencv2\opencv.hpp>

namespace Util
{
	//-! Convert cv::Mat to QIMage
	QImage Mat2QImage(const cv::Mat& mat); 

	//-! Convert cv::Mat to QPixmap
	QPixmap Mat2QPixmap(const cv::Mat& mat);

	//-! Convert QImage to cv::Mat
	cv::Mat QImage2Mat(const QImage& image);

	//-! Convert QPixmap to cv::Mat
	cv::Mat QPixmap2Mat(const QPixmap& pixMap);
}

#endif // !_MAT_2_QIMAGE_
