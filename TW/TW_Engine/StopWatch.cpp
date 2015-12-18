#include "StopWatch.h"

namespace TW
{
namespace Timing
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
		}
	}

} // namespace Timing
} // namespace TW
