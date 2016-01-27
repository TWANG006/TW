/************************************************************************/
/* qt-opencv-multithreaded:                                             */
/* A multithreaded OpenCV application using the Qt framework.           */
/*                                                                      */
/* SharedImageBuffer.h                                                  */
/*                                                                      */
/* Nick D'Ademo <nickdademo@gmail.com>                                  */
/*                                                                      */
/* Copyright (c) 2012-2015 Nick D'Ademo                                 */
/*                                                                      */
/* Permission is hereby granted, free of charge, to any person          */
/* obtaining a copy of this software and associated documentation       */
/* files (the "Software"), to deal in the Software without restriction, */
/* including without limitation the rights to use, copy, modify, merge, */
/* publish, distribute, sublicense, and/or sell copies of the Software, */
/* and to permit persons to whom the Software is furnished to do so,    */
/* subject to the following conditions:                                 */
/*                                                                      */
/* The above copyright notice and this permission notice shall be       */
/* included in all copies or substantial portions of the Software.      */
/*                                                                      */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,      */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF   */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                */
/* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS  */
/* BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN   */
/* ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN    */
/* CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     */
/* SOFTWARE.                                                            */
/*                                                                      */
/* Modified by WANG Tianyi(tianyiwang666@gmail.com)                     */
/*                                                                      */
/************************************************************************/

#ifndef SHAREDIMAGEBUFFER_H
#define SHAREDIMAGEBUFFER_H

#include <QHash>
#include <QSet>
#include <QWaitCondition>
#include <QMutex>

#include <opencv2/opencv.hpp>

#include "TW_Concurrent_Buffer.h"

namespace TW{

using ImageBuffer = std::shared_ptr<TW::Concurrent_Buffer<cv::Mat> >;
using BufferHash =  QHash<int, ImageBuffer>;

class TW_LIB_DLL_EXPORTS SharedImageBuffer
{
public:
	SharedImageBuffer();
	void add(int deviceNumber, const ImageBuffer &imageBuffer, bool sync = false);
	ImageBuffer getByDeviceNumber(int deviceNumber);
	void removeByDeviceNumber(int deviceNumber);
	void sync(int deviceNumber);
	void wakeAll();
	void setSyncEnabled(bool enable);
	bool isSyncEnabledForDeviceNumber(int deviceNumber);
	bool getSyncEnabled();
	bool containsImageBufferForDeviceNumber(int deviceNumber);

private:
	BufferHash m_imageBufferMap;
	QSet<int> m_syncSet;
	QWaitCondition m_wc;
	QMutex m_mutex;
	int m_nArrived;
	bool m_doSync;
};

} // namespace TW
#endif // SHAREDIMAGEBUFFER_H
