//---------------------------------------------------------------------------
//
// Project: OpenWalnut ( http://www.openwalnut.org )
//
// Copyright 2009 OpenWalnut Community, BSV@Uni-Leipzig and CNCF@MPI-CBS
// For more information see http://www.openwalnut.org/copying
//
// This file is part of OpenWalnut.
//
// OpenWalnut is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// OpenWalnut is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with OpenWalnut. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#ifndef WLPROFILERLOGGER_H_
#define WLPROFILERLOGGER_H_

#include <core/common/WLogger.h>

#include "WLProfiler.h"

/**
 * Logs profiler results.
 *
 * \author pieloth
 */
class WLProfilerLogger
{
public:
    /**
     * Returns an instance of WLProfilerLogger for logging.
     */
    static WLProfilerLogger& instance();

    /**
     * Overload for streamed output.
     *
     * \param profiler Profiler to log.
     */
    WLProfilerLogger& operator<<( WLProfiler const& profiler );

private:
    WLProfilerLogger();
    virtual ~WLProfilerLogger();
};

namespace wlprofiler
{
    /**
     * Global method to log profiler results.
     */
    WLProfilerLogger& log();
}

inline WLProfilerLogger& WLProfilerLogger::instance()
{
    // Destructor is called when application is being closed.
    static WLProfilerLogger instance;
    return instance;
}

inline WLProfilerLogger& WLProfilerLogger::operator <<( WLProfiler const& profiler )
{
    wlog::info( profiler.getName() ) << profiler;
    return *this;
}

inline WLProfilerLogger& wlprofiler::log()
{
    return WLProfilerLogger::instance();
}

#endif  // WLPROFILERLOGGER_H_
