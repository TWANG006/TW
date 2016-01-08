#include "TW_StopWatch.h"

namespace TW
{
	StopWatch::StopWatch()
		:m_startTime()
		, m_endTime()
		, m_dElapsedTime(0.0)
		, m_dTotalTime(0.0)
		, m_isRunning(false)
		, m_iNumClockSessions(0)
		, m_dFreq(0.0)
		, m_isFreqSet(false)
	{
		if (!m_isFreqSet)
		{
			LARGE_INTEGER temp;
			QueryPerformanceFrequency((LARGE_INTEGER*)&temp);
			m_dFreq = (static_cast<double>(temp.QuadPart)) / 1000.0;
			m_isFreqSet = true;
		}
	}

	StopWatch::~StopWatch()
	{

	}

	void StopWatch::start()
	{
		QueryPerformanceCounter((LARGE_INTEGER*)&m_startTime);
		m_isRunning = true;
	}

	void StopWatch::stop()
	{
		QueryPerformanceCounter((LARGE_INTEGER*)&m_endTime);
		m_dElapsedTime = (static_cast<double>(m_endTime.QuadPart - m_startTime.QuadPart)) / m_dFreq;

		m_dTotalTime += m_dElapsedTime;
		m_iNumClockSessions++;
		m_isRunning = false;
	}

	void StopWatch::reset()
	{
		m_dElapsedTime = 0.0;
		m_dTotalTime = 0.0;
		m_iNumClockSessions = 0;

		if (m_isRunning)
		{
			QueryPerformanceCounter((LARGE_INTEGER*)&m_startTime);
		}
	}

	double StopWatch::getElapsedTime()
	{
		double tempTotal = m_dTotalTime;

		if (m_isRunning)
		{
			LARGE_INTEGER temp;
			QueryPerformanceCounter((LARGE_INTEGER*)&temp);
			tempTotal += ((static_cast<double>(temp.QuadPart - m_startTime.QuadPart)) / m_dFreq);
		}

		return tempTotal;
	}

	double StopWatch::getAverageTime()
	{
		return (m_iNumClockSessions > 0) ? (m_dTotalTime / static_cast<double>(m_iNumClockSessions)) : 0.0;
	}

} // namespace TW
