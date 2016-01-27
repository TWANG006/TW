#ifdef TW_CONCURRENT_BUFFER_H
template<class T> Concurrent_Buffer<T>::Concurrent_Buffer()
	:m_size(1)
{
	// Allocate slots for semaphores
	m_freeSlotsSemaphore = new QSemaphore(m_size);
	m_occupiedSlotsSemaphore = new QSemaphore(0);
	m_enQueueSemaphore = new QSemaphore(1);
	m_deQueueSemaphore = new QSemaphore(1);
}

template<class T> Concurrent_Buffer<T>::Concurrent_Buffer(size_t size)
	:m_size(size)
{
	// Allocate slots for semaphores
	m_freeSlotsSemaphore = new QSemaphore(m_size);
	m_occupiedSlotsSemaphore = new QSemaphore(0);
	m_enQueueSemaphore = new QSemaphore(1);
	m_deQueueSemaphore = new QSemaphore(1);
}

template<class T> void Concurrent_Buffer<T>::EnQueue(const T& elem, bool dropIfFull)
{
	m_enQueueSemaphore->acquire();

	// If dropIfFull is true, do not block the block 
	if(dropIfFull)
	{
		// Use tryAcquire to not block the execution
		// Only acquire the resourse when there is an available slot
		if(m_freeSlotsSemaphore->tryAcquire())
		{
			m_mutex.lock();
			m_queue.enqueue(elem);
			m_mutex.unlock();

			m_occupiedSlotsSemaphore->release();
		}
	}

	// If dropIfFull is false, block the execution and wait until there is
	// a free slot
	else
	{
		m_freeSlotsSemaphore->acquire();
		
		m_mutex.lock();
		m_queue.enqueue(elem);
		m_mutex.unlock();

		m_occupiedSlotsSemaphore->release();
	}

	m_enQueueSemaphore->release();
}

template<class T> void Concurrent_Buffer<T>::DeQueue(T& elem)
{
	m_deQueueSemaphore->acquire();

	// Acquire the occupied semaphore
	m_occupiedSlotsSemaphore->acquire();

	m_mutex.lock();
	elem = m_queue.dequeue();
	m_mutex.unlock();

	// Release a free slot
	m_freeSlotsSemaphore->release();

	m_deQueueSemaphore->release();
}

template<class T> std::shared_ptr<T> Concurrent_Buffer<T>::DeQueue()
{
	m_deQueueSemaphore->acquire();

	// Acquire the occupied semaphore
	m_occupiedSlotsSemaphore->acquire();

	m_mutex.lock();
	std::shared_ptr<T> elem(std::make_shared<T>(m_queue.dequeue()));
	m_mutex.unlock();

	// Release a free slot
	m_freeSlotsSemaphore->release();

	m_deQueueSemaphore->release();

	return elem;
}

template<class T> bool Concurrent_Buffer<T>:: IsEmpty() const
{
	QMutexLocker mLocker(&m_mutex);
	return m_queue.isEmpty();
}

template<class T> bool Concurrent_Buffer<T>:: IsFull() const
{
	QMutexLocker mLocker(&m_mutex);
	return m_queue.size() == m_size;
}

template<class T> bool Concurrent_Buffer<T>:: clear()
{
	if(m_queue.size() > 0)
	{
		if(m_enQueueSemaphore->tryAcquire())
		{
			if(m_deQueueSemaphore->tryAcquire())
			{
				m_freeSlotsSemaphore->release(m_queue.size());
				m_freeSlotsSemaphore->acquire(m_size);
				m_occupiedSlotsSemaphore->acquire(m_queue.size());
				m_queue.clear();
				m_freeSlotsSemaphore->release(m_size);
				m_deQueueSemaphore->release();
			}
			else
			{
				return false;
			}

			m_enQueueSemaphore->release();
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}
}
#endif