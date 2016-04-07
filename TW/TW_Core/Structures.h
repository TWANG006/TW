#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <TW_Concurrent_Buffer.h>
#include <TW_paDIC_cuFFTCC2D.h>
#include <opencv2\opencv.hpp>

#include <QRect>
#include <QOpenGLcontext>
#include <QOpenGLBuffer>
#include <QOpenGLtexture>
#include <QSurface>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_3_3_Core>
#include <QMutex>

#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

template<typename T>
void deleteObject(T*& param)
{
	if (param != nullptr && param !=0 && param !=NULL)
	{
		delete param;
		param = nullptr;
	}
}

struct SharedResources{
	QOpenGLContext *sharedContext;
	QOpenGLBuffer *sharedVBO;
	QOpenGLTexture *sharedTexture;
	QOpenGLShaderProgram *sharedProgram;
	QSurface *sharedSurface;

	cudaArray *cudaImgArray;		// CUDA texture array
	cudaGraphicsResource *cuda_ImgTex_Resource;

	SharedResources()
		: sharedContext(nullptr)
		, sharedVBO(nullptr)
		, sharedTexture(nullptr)
		, sharedProgram(nullptr)
		, sharedSurface(nullptr)
		, cudaImgArray(nullptr)
		, cuda_ImgTex_Resource(nullptr)
	{}
};

using cuFftcc2D = TW::paDIC::cuFFTCC2D;
using cuFftcc2DPtr = std::unique_ptr<cuFftcc2D>;
using ImageBuffer = TW::Concurrent_Buffer<cv::Mat>;
using ImageBufferPtr = std::shared_ptr<ImageBuffer>;

typedef struct 
{
	bool m_isLeftBtnReleased;
	bool m_isRightBtnReleased;
	QRect m_roiBox;
} MouseData;

typedef struct
{
    int averageFPS;
    int nFramesProcessed;
} ThreadStatisticsData;

#endif // STRUCTURES_H