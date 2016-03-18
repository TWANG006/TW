#ifndef TW_CUTEMPUTILS_H
#define TW_CUTEMPUTILS_H


namespace TW{
// ------------------------------------CUDA template utilities------------------------------!
template<typename T>
void cuInitialize(T* devPtr, const T val, const size_t nwords, int devID);
}

#endif // !TW_CUTEMPUTILS_H
