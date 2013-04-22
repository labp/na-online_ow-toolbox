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

#include <list>
#include <string>

#include "core/common/WLogger.h"
#include "core/common/WRealtimeTimer.h"

#include "WLTimeProfiler.h"

using std::string;
using std::list;

namespace LaBP
{
    const string WLTimeProfiler::CLASS = "WLTimeProfiler";

    const double WLTimeProfiler::NO_TIME = -1.0;

    WLTimeProfiler::WLTimeProfiler( string clazz, string action ) :
                    m_class( clazz ), m_action( action ), m_isStarted( false ), m_isStopped( false ), m_start( 0 ), m_stop( 0 )
    {
        m_timer.reset( new WRealtimeTimer );
    }

    WLTimeProfiler::WLTimeProfiler( const WLTimeProfiler& profiler )
    {
        m_class = profiler.m_class;
        m_action = profiler.m_action;

        m_start = profiler.m_start;
        m_isStarted = profiler.m_isStarted;

        m_stop = profiler.m_stop;
        m_isStopped = profiler.m_isStopped;

        m_profilers = profiler.m_profilers;

        m_timer.reset( new WRealtimeTimer );
    }

    WLTimeProfiler::~WLTimeProfiler()
    {
    }

    string WLTimeProfiler::getClass() const
    {
        return m_class;
    }

    void WLTimeProfiler::addChild( WLTimeProfiler::SPtr profiler )
    {
        m_profilers.push_back( profiler );
    }

    WLTimeProfiler::SPtr WLTimeProfiler::createAndAdd( string clazz, string action )
    {
        WLTimeProfiler::SPtr profiler( new WLTimeProfiler( clazz, action ) );
        m_profilers.push_back( profiler );
        return profiler;
    }

    string WLTimeProfiler::getAction() const
    {
        return m_action;
    }

    list< WLTimeProfiler::SPtr >& WLTimeProfiler::getProfilers()
    {
        return m_profilers;
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
        m_timer->reset();
        m_isStarted = true;
        return m_start;
    }

    bool WLTimeProfiler::isStarted()
    {
        return m_isStarted;
    }

    WLTimeProfiler::TimeT WLTimeProfiler::stop()
    {
        m_stop = m_timer->elapsed() * 1000;
        m_isStopped = true;
        return m_stop;
    }

    bool WLTimeProfiler::isStopped()
    {
        return m_isStopped;
    }

    void WLTimeProfiler::log()
    {
        wlog::info( WLTimeProfiler::CLASS ) << m_class << "::" << m_action << ": " << getMilliseconds() << " ms";
    }

    void WLTimeProfiler::stopAndLog()
    {
        stop();
        log();
    }

    WLTimeProfiler::SPtr WLTimeProfiler::clone() const
    {
        WLTimeProfiler::SPtr profiler( new WLTimeProfiler( *this ) );
        return profiler;
    }

}  // namespace LaBP
