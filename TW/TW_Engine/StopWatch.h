#ifndef ENGINE_STOPWATCH_H
#define ENGINE_STOPWATCH_H

#include <Windows.h>

namespace TW
{
	namespace Timing
	{

		class __declspec(dllexport) StopWatch
		{
		public:
			StopWatch();
			~StopWatch();

			void start();				//! Start timer
			void stop();				//! Stop timer
			void reset();				//! Reset timer

			//! Get elapsed time after calling start() or time between
			//! stop() and start()
			double getElapsedTime();

			//! Average time = TotalTime / number of stops
			double getAverageTime();

		private:
			LARGE_INTEGER m_startTime;	//! Start time
			LARGE_INTEGER m_endTime;	//! End time

			double m_dElapsedTime;		//! Time elapsed between the last start and stop
			double m_dTotalTime;			//! Time elapsed between all starts and stops
			bool m_isRunning;			//! Judge if timer is still running
			int m_iNumClockSessions;	//! Number of starts and stops
			double m_dFreq;				//! Frequency
			bool m_isFreqSet;			//! Judge if the frequency is set 
		};

	}	// namespace Timing
}	// namespace TW

#endif // !ENGINE_STOPWATCH_H
