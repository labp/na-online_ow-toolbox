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

#ifndef WLLIFETIMEPROFILER_TEST_H_
#define WLLIFETIMEPROFILER_TEST_H_

#include <cxxtest/TestSuite.h>

#include "../WLLifetimeProfiler.h"

class WLLifetimeProfilerTest: public CxxTest::TestSuite
{
public:
    void test_clone()
    {
        WLLifetimeProfiler lp0( "test", "test" );
        usleep( 50000 );
        TS_ASSERT_EQUALS( lp0.getCloneCounter(), 0 );
        TS_ASSERT_LESS_THAN( 40, lp0.getAge() );

        WLLifetimeProfiler lp1( lp0 );
        usleep( 50000 );
        TS_ASSERT_EQUALS( lp1.getCloneCounter(), 1 );
        TS_ASSERT_LESS_THAN( 90, lp1.getAge() );

        lp0.pause();
        lp1.pause();
        TS_ASSERT_LESS_THAN( 90, lp1.getAge() );

        WLLifetimeProfiler lp2( lp1 );
        usleep( 50000 );
        TS_ASSERT_EQUALS( lp2.getCloneCounter(), 2 );
        TS_ASSERT_LESS_THAN( 140, lp2.getAge() );

        lp2.pause();
        TS_ASSERT_LESS_THAN( 140, lp2.getAge() );

        TS_ASSERT_EQUALS( lp0.getCloneCounter(), 0 );
        TS_ASSERT_LESS_THAN( lp0.getAge(), 140 );

        TS_ASSERT_EQUALS( lp1.getCloneCounter(), 1 );
        TS_ASSERT_LESS_THAN( lp1.getAge(), 140 );
    }
};

#endif  // WLLIFETIMEPROFILER_TEST_H_
