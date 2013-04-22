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

#include <list>
#include <string>

#include <boost/shared_ptr.hpp>

#include "core/common/WTimer.h"

using std::string;
using std::list;

namespace LaBP
{
    /**
     * Explicit time measurement with start and stop method. An instance relates to a class and an action.
     */
    class WLTimeProfiler
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
        WLTimeProfiler( string clazz, string action );

        WLTimeProfiler( const WLTimeProfiler& profiler );

        virtual ~WLTimeProfiler();

        /**
         * Returns the class.
         *
         * \return class name
         */
        string getClass() const;

        /**
         * Returns the action name.
         *
         * \return action name
         */
        string getAction() const;

        /**
         * Adds a time profiler to this object, i.e. to store previous measurements.
         *
         * \param profiler Measurement to add
         */
        void addChild( WLTimeProfiler::SPtr profiler );

        /**
         * Creates a new measurement object and add it to children list.
         *
         * \param clazz Class to measure
         * \param action Action or method to measure
         *
         * \return Create object
         */
        WLTimeProfiler::SPtr createAndAdd( string clazz, string action );

        /**
         * Returns the children profiler.
         *
         * \return list of profiler
         */
        list< WLTimeProfiler::SPtr >& getProfilers();

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
         * Logs the status as info: <clazz>::<action>: <milliseconds> ms
         */
        void log();

        /**
         * Stops the measurement and logs the status as info.
         */
        void stopAndLog();

        /**
         * Creates a shallow copy: no children
         *
         * \return new profiler instance
         */
        WLTimeProfiler::SPtr clone() const;

    private:
        list< WLTimeProfiler::SPtr > m_profilers;

        WTimer::SPtr m_timer;

        string m_class;
        string m_action;

        TimeT m_start;
        bool m_isStarted;

        TimeT m_stop;
        bool m_isStopped;
    };

} // namespace LaBP
#endif  // WLTIMEPROFILER_H_
