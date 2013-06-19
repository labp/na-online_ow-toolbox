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

#ifndef WLTIMEPROFILER_H_
#define WLTIMEPROFILER_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/WRealtimeTimer.h>

#include "WLProfiler.h"

using std::string;

/**
 * Explicit time measurement with start and stop method. An instance relates to a class and an action.
 */
class WLTimeProfiler: public WLProfiler
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLTimeProfiler > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLTimeProfiler > ConstSPtr;

    typedef double TimeT;

    static const string CLASS;

    static const double NO_TIME;

    /**
     * Constructor to measure an action or method of a class.
     *
     * \param clazz Class to measure
     * \param action Action or method to measure
     */
    WLTimeProfiler( string clazz, string action, bool autoLog = true );

    WLTimeProfiler( const WLTimeProfiler& profiler );

    virtual ~WLTimeProfiler();

    virtual std::ostream& write( std::ostream& strm ) const;

    virtual std::string getName() const;

    /**
     * Returns the start time.
     *
     * \return start time
     */
    TimeT getStart() const;

    /**
     * Returns the stop time.
     *
     * \return stop time
     */
    TimeT getStop() const;

    /**
     * Returns the time difference between start and stop in milliseconds or NO_TIME.
     *
     * \return difference between start/stop or NO_TIME if measurement is not started or stopped.
     */
    double getMilliseconds() const;

    /**
     * Sets a user defined time in milliseconds.
     */
    void setMilliseconds( double ms );

    /**
     * Starts a measurement.
     *
     * \return time stamp when the measurement was started.
     */
    TimeT start();

    /**
     * Indicates if the measurement is started.
     *
     * \return true if measurement is started.
     */
    bool isStarted();

    /**
     * Stops a measurement.
     *
     * \return time stamp when the measurement was stopped.
     */
    TimeT stop();

    /**
     * Indicates if the measurement is stopped.
     *
     * \return true if measurement is stopped.
     */
    bool isStopped();

    /**
     * Creates a shallow copy: no children
     *
     * \return new profiler instance
     */
    WLTimeProfiler::SPtr clone() const;

private:
    WRealtimeTimer m_timer;

    TimeT m_start;
    bool m_isStarted;

    TimeT m_stop;
    bool m_isStopped;
};

#endif  // WLTIMEPROFILER_H_
