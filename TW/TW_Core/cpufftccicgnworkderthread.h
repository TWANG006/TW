#ifndef CPUFFTCCICGNWORKDERTHREAD_H
#define CPUFFTCCICGNWORKDERTHREAD_H

#include <QObject>
#include "Structures.h"

#include "TW.h"
#include "TW_paDIC_FFTCC2D_CPU.h"
#include "TW_paDIC_ICGN2D_CPU.h"

class CPUFFTCCICGNWorkderThread : public QObject
{
	Q_OBJECT

public:
	CPUFFTCCICGNWorkderThread() = delete;
	CPUFFTCCICGNWorkderThread(const CPUFFTCCICGNWorkderThread&) = delete;
	CPUFFTCCICGNWorkderThread& operator=(const CPUFFTCCICGNWorkderThread&) = delete;

	CPUFFTCCICGNWorkderThread(//Inputs
						      ImageBufferPtr refImgBuffer,
							  ImageBufferPtr tarImgBuffer,
							  int iWidth, int iHeight,
							  int iSubsertX, int iSubsetY,
							  int iGridSpaceX, int iGridSpaceY,
							  int iMarginX, int iMarginY,
							  const QRect &roi,
							  const cv::Mat &firstFrame);

	~CPUFFTCCICGNWorkderThread();

private:
	
};

#endif // CPUFFTCCICGNWORKDERTHREAD_H
