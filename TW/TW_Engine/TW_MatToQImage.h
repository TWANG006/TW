#ifndef TW_MATTOQIMAGE_H
#define TW_MATTOQIMAGE_H

#include <QImage>
#include <QPixmap>
#include <QDebug>
#include <opencv2\opencv.hpp>

#include "TW.h"

namespace TW{

//-! Convert cv::Mat to QIMage
TW_LIB_DLL_EXPORTS QImage Mat2QImage(const cv::Mat& mat); 

//-! Convert cv::Mat to QPixmap
TW_LIB_DLL_EXPORTS QPixmap Mat2QPixmap(const cv::Mat& mat);

//-! Convert QImage to cv::Mat
TW_LIB_DLL_EXPORTS cv::Mat QImage2Mat(const QImage& image);

//-! Convert QPixmap to cv::Mat
TW_LIB_DLL_EXPORTS cv::Mat QPixmap2Mat(const QPixmap& pixMap);

} //!- namespace TW

#endif // !TW_MATTOQIMAGE_H
