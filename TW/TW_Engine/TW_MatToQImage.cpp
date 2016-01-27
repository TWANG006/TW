#include "TW_MatToQImage.h"

namespace TW{

QImage Mat2QImage(const cv::Mat& mat)
{
	switch (mat.type())
	{
		//-! 8-bit, 4 channel
		case CV_8UC4:
		{
			QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB32);
			return image;
		}

		//-! 8-bit, 3 channel
		case CV_8UC3:
		{
			QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
			return image;
		}

		//-! 8-bit, 1 channel
		case CV_8UC1:
		{
			//-! Only construct the LUT onece
			static QVector<QRgb> vColorTable;

			if (vColorTable.isEmpty())
			{
				for (int i = 0; i < 256; ++i)
				{
					vColorTable.push_back(qRgb(i, i, i));
				}
			}

			QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
			image.setColorTable(vColorTable);
			return image;
		}
		default:
			qCritical() << QString("Error!! Cannot convert cv::Mat to QImage "
				"with the input image type: %1").arg(mat.type());
			break;
	}

	return QImage();
}

QPixmap Mat2QPixmap(const cv::Mat& mat)
{
	return QPixmap::fromImage(Mat2QImage(mat));
}

cv::Mat QImage2Mat(const QImage& image)
{
	switch (image.format())
	{
		//-! 8-bit, 4 channels
	case QImage::Format_RGB32:
	{
		cv::Mat mat(image.height(), image.width(),
			CV_8UC4, const_cast<uchar*>(image.bits()), image.bytesPerLine());

		return mat.clone();
	}

	//!- 8-bit, 3 channels
	case QImage::Format_RGB888:
	{
		QImage swapped = image.rgbSwapped();

		return cv::Mat(swapped.height(), swapped.width(),
			CV_8UC3, const_cast<uchar*>(swapped.bits()), swapped.bytesPerLine()).clone();
	}

	//!- 8-bit, 1 channel
	case QImage::Format_Indexed8:
	{
		cv::Mat mat(image.height(), image.width(),
			CV_8UC1, const_cast<uchar*>(image.bits()), image.bytesPerLine());

		return mat.clone();
	}
	default:
		qCritical() << QString("Error!! Cannot convert QImage to cv::Mat"
			"with the input image type: %1").arg(image.format());
		break;
	}
	return cv::Mat();
}

cv::Mat QPixmap2Mat(const QPixmap& pixMap)
{
	return QImage2Mat(pixMap.toImage());
}

} //!- namespace TW