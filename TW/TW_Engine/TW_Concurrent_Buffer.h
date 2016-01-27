/* Author: Nick D'Ademo <nickdademo@gmail.com>                          */
/*                                                                      */
/* Copyright (c) 2012-2016 Nick D'Ademo                                 */
/*                                                                      */
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
/* Modified by WANG Tianyi <tianyiwang666@gmail.com>                    */
/************************************************************************/

#ifndef TW_CONCURRENT_BUFFER_H
#define TW_CONCURRENT_BUFFER_H

#include "TW.h"
#include <QMutex>
#include <QSemaphore>
#include <QQueue>

#include <memory>

namespace TW{

/// \brief A self-implemented concurrent (circular) template container class for multithreading codes.
/// TODO: Improve 
/// NOTE: Some operations of this class, e.g. unsafe_size() is not thread-safe.
///
template<class T> class Concurrent_Buffer
{
public:

	/// \brief Default constructor. The buffer size is set to 1
	Concurrent_Buffer();

	/// \brief Overloaded constructor. Initialize the buffer with size size 
	///
	/// \param size the size of the buffer
	Concurrent_Buffer(size_t size);

	/// \brief thread-safe enQueue method
	///
	/// \ param elem the element to be enqueued
	/// \ param dropIfFull set to true to drop frames if the buffer is full
	void EnQueue(const T& elem, bool dropIfFull = false);

	/// \brief thread-safe deQueue method, the dequeued element is passed back by 
	/// reference
	///
	/// \ param elem the element to be dequeued
	void DeQueue(T& elem);

	/// \brief thread-safe deQueue method, the dequeued element is returned to a 
	/// smart pointer. nullptr is returned if there is no element in the queue
	std::shared_ptr<T> DeQueue();

	/// \brief unthread-safe method to get the "current" size of the queue
	inline size_t Usafe_CurrentSize() const	{ return m_queue.size(); }
	/// \brief get the maximum size of the buffer
	inline size_t Size() const { return m_size; }

	/// \brief Return true if no element is currently in the buffer
	bool IsEmpty() const;

	/// \brief Return true if the buffer is full
	bool IsFull() const;

	/// \brief Return ture is the Buffer is successfully cleared
	bool clear();

private:
	mutable QMutex m_mutex;
	QQueue<T> m_queue;
	QSemaphore *m_freeSlotsSemaphore;
	QSemaphore *m_occupiedSlotsSemaphore;
	QSemaphore *m_enQueueSemaphore;
	QSemaphore *m_deQueueSemaphore;

	size_t m_size;
};

#include "TW_Concurrent_Buffer.cpp"
}

#endif // !TW_CONCURRENT_BUFFER_H
