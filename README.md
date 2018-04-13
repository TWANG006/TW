# TW
Parallel DIC Engine and its application research codes.

## Introduction
This repository provides a pipelined real-time DIC system implementation that unifying the computation capabilities of both CPU and GPU. The system framework is based on the figure below: 
![RT-DIC System Framework](/imgs/Variant2.jpg)

1. TW_Engine: The revised implementations of the paDIC algorithm to make thame more suitable for real-time systems and applications
2. TW_EngineTester: Use Google Test to perform the unit test of the TW_Engine
3. TW_Core: The implementation of the proposed real-time DIC system.

## Dependencies
*Note: Please make sure you have at least one camera connected to the computer before you start.

1. [Intel Math Kernel Library (MKL)](https://software.intel.com/en-us/performance-libraries): using fftw3 to do fast Fourier transform (FFT) and LAPACK routine to solve linear system in parlalel on CPU.
2. [CUDA 8.0+](https://developer.nvidia.com/cuda-80-ga2-download-archive): for parallel computing on NVIDIA GPUs.
3. CUFFT: associated with CUDA, for perform parallel FFT on GPU.
4. [Qt 5.5+ with OpenGL Integration](https://www1.qt.io/qt5-5/): for GUI and multi-media used in App_DPRA.
5. [OpenCV 3.1+](https://opencv.org/opencv-3-1.html): for fast and convenient image I/O. 

## References
[[1] Zhang, L., Wang, T., Jiang, Z., Kemao, Q., Liu, Y., Liu, Z., ... & Dong, S. (2015). High accuracy digital image correlation powered by GPU-based parallel computing. Optics and Lasers in Engineering, 69, 7-12.](https://www.sciencedirect.com/science/article/pii/S0143816615000135)

[[2] Wang, T., Jiang, Z., Kemao, Q., Lin, F., & Soon, S. H. (2016). GPU accelerated digital volume correlation. Experimental Mechanics, 56(2), 297-309.](https://link.springer.com/article/10.1007/s11340-015-0091-4)

[[3]] Wang, T., Kemao, Q., Lin, F., Seah, HS. (2018) A Flexible Hegerogeneous Real-Time Digital Image Correlation System, Optics and Lasers in Engineering, Peer Review.
