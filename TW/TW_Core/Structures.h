#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <TW_Concurrent_Buffer.h>
#include <TW_paDIC_cuFFTCC2D.h>
#include <opencv2\opencv.hpp>

#include <QRect>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLContext>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <memory>
#include <mutex>

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

///\brief Implement the global intances as a singleton. SharedResources stores all 
/// the needed shared resources used by OpenGL widget and CUDA-openGL interoperability
/// utilities.
class SharedResources
{
public:
	static SharedResources& GetIntance();
	virtual ~SharedResources();

	SharedResources(const SharedResources&) = delete;
	SharedResources& operator=(const SharedResources&) = delete;
	
protected:
	SharedResources();

private:
	static std::unique_ptr<SharedResources> g_instance;	// The static global instance
	static std::once_flag m_onceFlag;
	
	//----Shared Resources below this point
	// Shared resources in OpenGL context
	QOpenGLContext		 *m_sharedContext;
	QOpenGLTexture		 *m_sharedTarImgBuffer;
	QOpenGLShaderProgram *m_sharedShaderProgram;

	// Shared resources for CUDA and openGL interoperability
	struct cudaGraphicsResource *m_cudaTexResource;
};

#endif // STRUCTURES_H