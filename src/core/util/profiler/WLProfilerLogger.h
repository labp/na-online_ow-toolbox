//---------------------------------------------------------------------------
//
// Project: NA-Online ( http://www.labp.htwk-leipzig.de )
//
// Copyright 2010 Laboratory for Biosignal Processing, HTWK Leipzig, Germany
//
// This file is part of NA-Online.
//
// NA-Online is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// NA-Online is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with NA-Online. If not, see <http://www.gnu.org/licenses/>.
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
 * \ingroup util
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
