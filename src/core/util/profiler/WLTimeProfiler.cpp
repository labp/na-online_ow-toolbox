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

#include "WLProfilerLogger.h"
#include "WLTimeProfiler.h"

using std::string;
using std::list;

const string WLTimeProfiler::CLASS = "WLTimeProfiler";

WLTimeProfiler::WLTimeProfiler( string clazz, string action, bool autoLog ) :
                WLProfiler( clazz, action, autoLog ), m_isStarted( false ), m_isStopped( false ), m_elapsed( 0 )
{
    if( autoLog )
    {
        this->start();
    }
}

WLTimeProfiler::WLTimeProfiler( const WLTimeProfiler& profiler ) :
                WLProfiler( profiler.m_source, profiler.m_action, profiler.m_autoLog )
{
    m_isStarted = profiler.m_isStarted;
    m_isStopped = profiler.m_isStopped;
    m_elapsed = profiler.getMilliseconds();
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

double WLTimeProfiler::getMilliseconds() const
{
    // Measurement is running
    if( m_isStarted == true && m_isStopped == false )
    {
        return m_elapsed + m_timer.elapsed() * 1000;
    }
    // Measurement is stopped
    if( m_isStarted == false && m_isStopped == true )
    {
        return m_elapsed;
    }
    // Measurement was not started, but maybe a time is set
    return m_elapsed;
}

void WLTimeProfiler::setMilliseconds( double ms )
{
    m_elapsed = ms;

    m_isStarted = false;
    m_isStopped = true;
}

WLTimeProfiler::TimeT WLTimeProfiler::start( bool reset )
{
    if( reset )
    {
        m_elapsed = 0;
    }
    m_timer.reset();
    m_isStarted = true;
    m_isStopped = false;
    return m_elapsed;
}

WLTimeProfiler::TimeT WLTimeProfiler::stop()
{
    if( m_isStarted )
    {
        m_elapsed += m_timer.elapsed() * 1000;
    }
    m_isStarted = false;
    m_isStopped = true;
    return m_elapsed;
}
