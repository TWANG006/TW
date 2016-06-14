#include "glwidget.h"

#include <vector>

GLWidget::GLWidget(std::shared_ptr<SharedResources> &s, 
				   QThread *&t, 
				   QWidget *parent,
				   int iWidth,
				   int iHeight,
				   const QRect &roi)
	: QOpenGLWidget(parent)
	, m_sharedResources(s)
	, m_renderThread(t)
	, m_viewWidth(0)
	, m_viewHeight(0)
	, m_iImgWidth(iWidth)
	, m_iImgHeight(iHeight)
	, m_ROI(roi)
{
	setMinimumSize(640, 960);
}

GLWidget::~GLWidget()
{
	makeCurrent();

	m_vao.destroy();

	deleteObject(m_sharedResources->sharedTexture);
	deleteObject(m_sharedResources->sharedProgram);
	m_sharedResources->sharedVBO->destroy();
	deleteObject(m_sharedResources->sharedVBO);
	
	doneCurrent();
}

void GLWidget::initializeGL()
{
	m_viewWidth = width();
	m_viewHeight = height();

	m_sharedResources->sharedContext = new QOpenGLContext;
	QOpenGLContext *shareContext = context();
	m_sharedResources->sharedContext->setFormat(shareContext->format());
	m_sharedResources->sharedContext->setShareContext(shareContext);
	m_sharedResources->sharedContext->create();
	m_sharedResources->sharedSurface = shareContext->surface();
	m_sharedResources->sharedContext->moveToThread(m_renderThread);

	initializeOpenGLFunctions();

	glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
	glClearDepth(1.0f);

	glDisable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
	glEnable(GL_TEXTURE_2D);


	//!- Compile Shaders
	m_sharedResources->sharedProgram = new QOpenGLShaderProgram();
	m_sharedResources->sharedProgram ->create();
	m_sharedResources->sharedProgram ->addShaderFromSourceFile(
		QOpenGLShader::Vertex, 
		QLatin1String("vert.glsl"));
	m_sharedResources->sharedProgram ->addShaderFromSourceFile(
		QOpenGLShader::Fragment,
		QLatin1String("frag.glsl"));

	m_sharedResources->sharedProgram ->link();

	//!- Create & bind VBO
	m_sharedResources->sharedVBO = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
	m_sharedResources->sharedVBO->create();
	m_sharedResources->sharedVBO->bind();

	static const GLfloat quad_data[] =
	{
		-1.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, -1.0f,
		-1.0f, -1.0f,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f
	};

	m_sharedResources->sharedVBO->allocate(quad_data, sizeof(quad_data));
	m_sharedResources->sharedVBO->setUsagePattern(QOpenGLBuffer::StaticDraw);
	m_sharedResources->sharedVBO->release();

	m_vao.create();
	if (m_vao.isCreated())
		m_vao.bind();
	m_sharedResources->sharedProgram ->bind();
	m_sharedResources->sharedVBO->bind();
	m_sharedResources->sharedProgram->setAttributeBuffer(
		0,
		GL_FLOAT,
		0,
		2,
		0);
	m_sharedResources->sharedProgram->enableAttributeArray(0);

	m_sharedResources->sharedProgram->setAttributeBuffer(
		1,
		GL_FLOAT,
		8 * sizeof(float),
		2,
		0);
	m_sharedResources->sharedProgram->enableAttributeArray(1);
	m_sharedResources->sharedVBO->release();
	m_sharedResources->sharedProgram->release();
	m_vao.release();

	//!- Create & bind ROI VBO
	m_sharedResources->sharedROIVBO = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
	m_sharedResources->sharedROIVBO->create();
	m_sharedResources->sharedROIVBO->bind();

	GLfloat tempX = 2.0f / (GLfloat)m_iImgWidth * m_ROI.x() - 1;
	GLfloat tempY = 2.0f / (GLfloat)m_iImgHeight * m_ROI.y() - 1;
	GLfloat tempROIX = 2.0f / (GLfloat)m_iImgWidth * (m_ROI.x() + m_ROI.width() - 1) - 1;
	GLfloat tempROIY = 2.0f / (GLfloat)m_iImgHeight *(m_ROI.y() + m_ROI.height() - 1) - 1;

	tempY = -tempY;
	tempROIY = -tempROIY;


	float ROI_data[] =
	{
		tempX, tempROIY,
		tempROIX, tempROIY,
		tempROIX, tempY,
		tempX, tempY,

		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f
	};

	m_sharedResources->sharedROIVBO->allocate(ROI_data, sizeof(ROI_data));
	m_sharedResources->sharedROIVBO->setUsagePattern(QOpenGLBuffer::StaticDraw);
	m_sharedResources->sharedROIVBO->release();

	m_ROIvao.create();
	if (m_ROIvao.isCreated())
		m_ROIvao.bind();
	m_sharedResources->sharedProgram ->bind();
	m_sharedResources->sharedROIVBO->bind();
	m_sharedResources->sharedProgram->setAttributeBuffer(
		0,
		GL_FLOAT,
		0,
		2,
		0);
	m_sharedResources->sharedProgram->enableAttributeArray(0);

	m_sharedResources->sharedProgram->setAttributeBuffer(
		1,
		GL_FLOAT,
		8 * sizeof(float),
		2,
		0);
	m_sharedResources->sharedProgram->enableAttributeArray(1);
	m_sharedResources->sharedROIVBO->release();
	m_sharedResources->sharedProgram->release();
	m_ROIvao.release();

	initGLTexture();
	initCUDAArray();
}

void GLWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0,m_viewHeight/2,m_viewWidth,m_viewHeight/2);

	m_sharedResources->sharedProgram->bind();
	m_vao.bind();
	m_sharedResources->sharedTexture->bind();

	glDrawArrays(GL_TRIANGLE_FAN,0,4);

	m_sharedResources->sharedTexture->release();
	m_vao.release();

	m_ROIvao.bind();
	m_sharedResources->sharedUTexture->bind();
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	m_sharedResources->sharedUTexture->release();
	m_sharedResources->sharedProgram->release();
	

	glViewport(0, 0,m_viewWidth,m_viewHeight/2);

	m_sharedResources->sharedProgram->bind();
	m_vao.bind();
	m_sharedResources->sharedTexture->bind();

	glDrawArrays(GL_TRIANGLE_FAN,0,4);

	m_sharedResources->sharedTexture->release();
	m_vao.release();

	m_ROIvao.bind();
	m_sharedResources->sharedVTexture->bind();
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	m_sharedResources->sharedVTexture->release();

	m_sharedResources->sharedProgram->release();
}

void GLWidget::resizeGL(int w, int h)
{
	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	//glEnable(GL_DEPTH_TEST);
	
	m_viewHeight = h;
	m_viewWidth = w;
}

void GLWidget::initCUDAArray()
{
	// Register textures in CUDA
	checkCudaErrors(cudaGraphicsGLRegisterImage(&m_sharedResources->cuda_ImgTex_Resource,
												m_sharedResources->sharedTexture->textureId(),
												GL_TEXTURE_2D,
												cudaGraphicsMapFlagsWriteDiscard));
	checkCudaErrors(cudaGraphicsMapResources(1, 
											 &m_sharedResources->cuda_ImgTex_Resource,
											 0));

	checkCudaErrors(cudaGraphicsGLRegisterImage(&m_sharedResources->cuda_U_Resource,
												m_sharedResources->sharedUTexture->textureId(),
												GL_TEXTURE_2D,
												cudaGraphicsMapFlagsWriteDiscard));
	checkCudaErrors(cudaGraphicsMapResources(1, 
											 &m_sharedResources->cuda_U_Resource,
											 0));

	checkCudaErrors(cudaGraphicsGLRegisterImage(&m_sharedResources->cuda_V_Resource,
												m_sharedResources->sharedVTexture->textureId(),
												GL_TEXTURE_2D,
												cudaGraphicsMapFlagsWriteDiscard));
	checkCudaErrors(cudaGraphicsMapResources(1, 
											 &m_sharedResources->cuda_V_Resource,
											 0));

	// Bind texutres to their respective CUDA arrays
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&m_sharedResources->cudaImgArray,
														  m_sharedResources->cuda_ImgTex_Resource,
														  0,
														  0));
	checkCudaErrors(cudaGraphicsUnmapResources(1,
											   &m_sharedResources->cuda_ImgTex_Resource,
											   0));

	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&m_sharedResources->cudaUArray,
														  m_sharedResources->cuda_U_Resource,
														  0,
														  0));
	checkCudaErrors(cudaGraphicsUnmapResources(1,
											   &m_sharedResources->cuda_U_Resource,
											   0));

	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&m_sharedResources->cudaVArray,
														  m_sharedResources->cuda_V_Resource,
														  0,
														  0));
	checkCudaErrors(cudaGraphicsUnmapResources(1,
											   &m_sharedResources->cuda_V_Resource,
											   0));
}

void GLWidget::initGLTexture()
{
	//1. Create Texture for the target Image
	m_sharedResources->sharedTexture = new QOpenGLTexture(QOpenGLTexture::Target2D);
	m_sharedResources->sharedTexture->create();
	m_sharedResources->sharedTexture->bind();
	m_sharedResources->sharedTexture->setSize(m_iImgWidth, m_iImgHeight);
	m_sharedResources->sharedTexture->setFormat(QOpenGLTexture::R8_UNorm);
	m_sharedResources->sharedTexture->allocateStorage(QOpenGLTexture::Red, QOpenGLTexture::UInt8);	

	m_sharedResources->sharedTexture->setSwizzleMask(
		QOpenGLTexture::RedValue,
		QOpenGLTexture::RedValue,
		QOpenGLTexture::RedValue,
		QOpenGLTexture::OneValue);
	m_sharedResources->sharedTexture->setMinMagFilters(QOpenGLTexture::Nearest, QOpenGLTexture::Nearest);
	m_sharedResources->sharedTexture->setWrapMode(QOpenGLTexture::DirectionS, QOpenGLTexture::ClampToEdge);
	m_sharedResources->sharedTexture->setWrapMode(QOpenGLTexture::DirectionT, QOpenGLTexture::ClampToEdge);
	m_sharedResources->sharedTexture->generateMipMaps();

	m_sharedResources->sharedTexture->release();

	//2. Create Textures for U color map
	m_sharedResources->sharedUTexture = new QOpenGLTexture(QOpenGLTexture::Target2D);
	m_sharedResources->sharedUTexture->create();
	m_sharedResources->sharedUTexture->bind();
	m_sharedResources->sharedUTexture->setSize(m_ROI.width(), m_ROI.height());
	m_sharedResources->sharedUTexture->setFormat(QOpenGLTexture::RGBA8_UNorm);
	m_sharedResources->sharedUTexture->allocateStorage(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8);	

	m_sharedResources->sharedUTexture->setSwizzleMask(
		QOpenGLTexture::RedValue,
		QOpenGLTexture::GreenValue,
		QOpenGLTexture::BlueValue,
		QOpenGLTexture::OneValue);
	m_sharedResources->sharedUTexture->setMinMagFilters(QOpenGLTexture::Nearest, QOpenGLTexture::Nearest);
	m_sharedResources->sharedUTexture->setWrapMode(QOpenGLTexture::DirectionS, QOpenGLTexture::ClampToEdge);
	m_sharedResources->sharedUTexture->setWrapMode(QOpenGLTexture::DirectionT, QOpenGLTexture::ClampToEdge);
	m_sharedResources->sharedUTexture->generateMipMaps();

	m_sharedResources->sharedUTexture->release();

	//3. Create Textures for V color map
	m_sharedResources->sharedVTexture = new QOpenGLTexture(QOpenGLTexture::Target2D);
	m_sharedResources->sharedVTexture->create();
	m_sharedResources->sharedVTexture->bind();
	m_sharedResources->sharedVTexture->setSize(m_ROI.width(), m_ROI.height());
	m_sharedResources->sharedVTexture->setFormat(QOpenGLTexture::RGBA8_UNorm);
	m_sharedResources->sharedVTexture->allocateStorage(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8);	

	m_sharedResources->sharedVTexture->setSwizzleMask(
		QOpenGLTexture::RedValue,
		QOpenGLTexture::GreenValue,
		QOpenGLTexture::BlueValue,
		QOpenGLTexture::OneValue);
	m_sharedResources->sharedVTexture->setMinMagFilters(QOpenGLTexture::Nearest, QOpenGLTexture::Nearest);
	m_sharedResources->sharedVTexture->setWrapMode(QOpenGLTexture::DirectionS, QOpenGLTexture::ClampToEdge);
	m_sharedResources->sharedVTexture->setWrapMode(QOpenGLTexture::DirectionT, QOpenGLTexture::ClampToEdge);
	m_sharedResources->sharedVTexture->generateMipMaps();

	m_sharedResources->sharedVTexture->release();
}