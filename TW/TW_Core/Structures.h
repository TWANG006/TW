#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <TW_Concurrent_Buffer.h>
#include <TW_paDIC_ICGN2D_CPU.h>
#include <TW_paDIC_cuFFTCC2D.h>
#include <TW_paDIC_cuICGN2D.h>
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


// FPS statistics queue lengths
#define PROCESSING_FPS_STAT_QUEUE_LENGTH    32

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
	QOpenGLBuffer *sharedROIVBO;
	QOpenGLTexture *sharedTexture;
	QOpenGLTexture *sharedUTexture;
	QOpenGLTexture *sharedVTexture;
	QOpenGLShaderProgram *sharedProgram;
	QSurface *sharedSurface;

	cudaArray *cudaImgArray;		// CUDA texture array
	cudaArray *cudaUArray;
	cudaArray *cudaVArray;
	cudaGraphicsResource *cuda_ImgTex_Resource;
	cudaGraphicsResource *cuda_U_Resource;
	cudaGraphicsResource *cuda_V_Resource;

	SharedResources()
		: sharedContext(nullptr)
		, sharedVBO(nullptr)
		, sharedROIVBO(nullptr)
		, sharedTexture(nullptr)
		, sharedUTexture(nullptr)
		, sharedVTexture(nullptr)
		, sharedProgram(nullptr)
		, sharedSurface(nullptr)
		, cudaImgArray(nullptr)
		, cudaUArray(nullptr)
		, cudaVArray(nullptr)
		, cuda_ImgTex_Resource(nullptr)
		, cuda_U_Resource(nullptr)
		, cuda_V_Resource(nullptr)
	{}
};

using cuFftcc2D = TW::paDIC::cuFFTCC2D;
using cuFftcc2DPtr = std::unique_ptr<cuFftcc2D>;

using cuICGN2D = TW::paDIC::cuICGN2D;
using cuICGN2DPtr = std::unique_ptr<cuICGN2D>;

using ICGN2D_CPU = TW::paDIC::ICGN2D_CPU;
using ICGN2DPtr = std::unique_ptr<ICGN2D_CPU>;

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

enum class ComputationMode
{
	GPUFFTCC,
	GPUFFTCC_ICGN,
	GPUFFTCC_CPUICGN
};

#endif // STRUCTURES_H