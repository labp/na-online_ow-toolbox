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

#include <string>

#include <core/common/WLogger.h>
#include <core/common/WRealtimeTimer.h>

#include "WLProfilerLogger.h"
#include "WLTimeProfiler.h"

using std::string;
using std::list;

const string WLTimeProfiler::CLASS = "WLTimeProfiler";

const double WLTimeProfiler::NO_TIME = -1.0;

WLTimeProfiler::WLTimeProfiler( string clazz, string action, bool autoLog ) :
                WLProfiler( clazz, action, autoLog ), m_isStarted( false ), m_isStopped( false ), m_start( 0 ), m_stop( 0 )
{
    this->start();
}

WLTimeProfiler::WLTimeProfiler( const WLTimeProfiler& profiler ) :
                WLProfiler( profiler.m_source, profiler.m_action )
{
    m_start = profiler.m_start;
    m_isStarted = profiler.m_isStarted;

    m_stop = profiler.m_stop;
    m_isStopped = profiler.m_isStopped;

    m_timer.reset();
}

WLTimeProfiler::~WLTimeProfiler()
{
    if( isAutoLog() && isStarted() )
    {
        stop();
        wlprofiler::log() << *this;
    }
}

std::ostream& WLTimeProfiler::write( std::ostream& strm ) const
{
    return strm << m_source << "::" << m_action << ": " << getMilliseconds() << " ms";
}

std::string WLTimeProfiler::getName() const
{
    return WLTimeProfiler::CLASS;
}

WLTimeProfiler::TimeT WLTimeProfiler::getStart() const
{
    return m_start;
}

WLTimeProfiler::TimeT WLTimeProfiler::getStop() const
{
    return m_stop;
}

double WLTimeProfiler::getMilliseconds() const
{
    if( m_isStarted == false || m_isStopped == false )
        return NO_TIME;
    return ( m_stop - m_start );
}

void WLTimeProfiler::setMilliseconds( double ms )
{
    m_start = 0;
    m_stop = ms;

    m_isStarted = true;
    m_isStopped = true;
}

WLTimeProfiler::TimeT WLTimeProfiler::start()
{
    m_start = 0;
    m_timer.reset();
    m_isStarted = true;
    return m_start;
}

bool WLTimeProfiler::isStarted()
{
    return m_isStarted;
}

WLTimeProfiler::TimeT WLTimeProfiler::stop()
{
    m_stop = m_timer.elapsed() * 1000;
    m_isStopped = true;
    return m_stop;
}

bool WLTimeProfiler::isStopped()
{
    return m_isStopped;
}

WLTimeProfiler::SPtr WLTimeProfiler::clone() const
{
    WLTimeProfiler::SPtr profiler( new WLTimeProfiler( *this ) );
    return profiler;
}
