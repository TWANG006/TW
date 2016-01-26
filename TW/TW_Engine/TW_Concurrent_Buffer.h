#ifndef TW_CONCURRENT_BUFFER_H
#define TW_CONCURRENT_BUFFER_H

#include "TW.h"
#include <QMutex>
#include <QSemaphore>
#include <QQueue>

namespace TW{

/// \brief A concurrent (circular) template container class for multithreading codes.
/// NOTE: Some operations of this class, e.g. unsafe_size() is not thread-safe.
///
template<class T> class Concurrent_Buffer
{
public:
	/// \brief 
	///
	/// \param size the size of the buffer
	Concurrent_Buffer();
	Concurrent_Buffer(size_t size);
};

}

#endif // !TW_CONCURRENT_BUFFER_H
