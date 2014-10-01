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

#ifndef WLTIMEPROFILER_TEST_H_
#define WLTIMEPROFILER_TEST_H_

#include <cxxtest/TestSuite.h>

#include "../WLTimeProfiler.h"

class WLTimeProfilerTest: public CxxTest::TestSuite
{
public:
    void test_isStartedStopped()
    {
        WLTimeProfiler tp1( "test", "test", false );
        TS_ASSERT( !tp1.isStarted() );
        TS_ASSERT( !tp1.isStopped() );

        tp1.start();
        TS_ASSERT( tp1.isStarted() );
        TS_ASSERT( !tp1.isStopped() );

        tp1.stop();
        TS_ASSERT( !tp1.isStarted() );
        TS_ASSERT( tp1.isStopped() );

        WLTimeProfiler tp2( "test", "test", false );
        tp2.setMilliseconds( 4 );
        TS_ASSERT( !tp1.isStarted() );
        TS_ASSERT( tp1.isStopped() );
    }

    void test_startStop()
    {
        WLTimeProfiler tp1( "test", "test", false );
        TS_ASSERT_EQUALS( tp1.start(), 0 );

        usleep( 50000 );
        TS_ASSERT_LESS_THAN( 40, tp1.getMilliseconds() );

        usleep( 50000 );
        WLTimeProfiler::TimeT time = tp1.stop();
        TS_ASSERT_LESS_THAN( 90, time )
        TS_ASSERT_EQUALS( time, tp1.getMilliseconds() );

        TS_ASSERT_EQUALS( time, tp1.start( false ) );

        usleep( 50000 );
        time = tp1.stop();
        TS_ASSERT_LESS_THAN( 140, time )
        TS_ASSERT_EQUALS( time, tp1.getMilliseconds() );

        TS_ASSERT_EQUALS( 0, tp1.start() );

        usleep( 50000 );
        time = tp1.stop();
        TS_ASSERT_LESS_THAN( 50, time )
        TS_ASSERT_EQUALS( time, tp1.getMilliseconds() );
    }

    void test_setMilliseconds()
    {
        WLTimeProfiler tp1( "test", "test", false );
        const WLTimeProfiler::TimeT init_ms = 10000;
        tp1.setMilliseconds( init_ms );

        TS_ASSERT_EQUALS( init_ms, tp1.start( false ) );
        usleep( 50000 );
        WLTimeProfiler::TimeT time = tp1.stop();
        TS_ASSERT_LESS_THAN( init_ms + 50, time )
        TS_ASSERT_EQUALS( time, tp1.getMilliseconds() );

        WLTimeProfiler tp2( "test", "test", false );
        tp2.setMilliseconds( init_ms );

        TS_ASSERT_EQUALS( init_ms, tp2.stop() );

        TS_ASSERT_EQUALS( init_ms, tp2.getMilliseconds() );
    }

    void test_copyConstructor()
    {
        WLTimeProfiler tp_org( "test", "test", false );
        tp_org.start();
        usleep( 50000 );
        tp_org.stop();

        WLTimeProfiler tp1( tp_org );
        TS_ASSERT_LESS_THAN( 40, tp1.getMilliseconds() );

        tp_org.start();
        usleep( 50000 );
        WLTimeProfiler tp2( tp_org );
        TS_ASSERT( tp2.isStarted() );
        usleep( 50000 );
        TS_ASSERT_LESS_THAN( 90, tp2.getMilliseconds() );
    }
};

#endif  // WLTIMEPROFILER_TEST_H_
