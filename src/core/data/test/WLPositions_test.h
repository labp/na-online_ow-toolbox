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

#ifndef WLPOSITIONS_TEST_H_
#define WLPOSITIONS_TEST_H_

#include <cxxtest/TestSuite.h>

#include <core/common/WLogger.h>
#include <core/common/exceptions/WOutOfBounds.h>
#include <core/common/exceptions/WPreconditionNotMet.h>

#include "../WLPositions.h"

class WLPositionTest: public CxxTest::TestSuite
{
public:
    void setUp( void )
    {
        WLogger::startup();
    }

    void test_appendFailCoordSystem( void )
    {
        WLPositions pos1;
        pos1.coordSystem( WLECoordSystem::HEAD );
        pos1.resize( 2 );

        WLPositions pos2;
        pos2.coordSystem( WLECoordSystem::DEVICE );
        pos2.resize( 2 );

        TS_ASSERT_THROWS( pos1 += pos2, WPreconditionNotMet );
    }

    void test_appendFailUnit( void )
    {
        WLPositions pos1;
        pos1.unit( WLEUnit::METER );
        pos1.resize( 2 );

        WLPositions pos2;
        pos2.unit( WLEUnit::VOLT );
        pos2.resize( 2 );

        TS_ASSERT_THROWS( pos1 += pos2, WPreconditionNotMet );
    }

    void test_appendFailExponent( void )
    {
        WLPositions pos1;
        pos1.unit( WLEUnit::METER );
        pos1.exponent( WLEExponent::MILLI );
        pos1.resize( 2 );

        WLPositions pos2;
        pos2.unit( WLEUnit::METER );
        pos2.exponent( WLEExponent::CENTI );
        pos2.resize( 2 );

        TS_ASSERT_THROWS( pos1 += pos2, WPreconditionNotMet );
    }

    void test_appendOk( void )
    {
        const WLPositions::IndexT n1 = 3;
        const WLPositions::IndexT n2 = 5;
        const WLPositions::PositionsT d1 = WLPositions::PositionsT::Random( 3, n1 );
        const WLPositions::PositionsT d2 = WLPositions::PositionsT::Random( 3, n2 );

        WLPositions pos1;
        pos1.data( d1 );

        WLPositions pos2;
        pos2.data( d2 );

        pos1 += pos2;

        TS_ASSERT_EQUALS( pos1.size(), n1 + n2 );

        TS_ASSERT( pos1.data().leftCols( n1 ) == d1 );
        TS_ASSERT( pos1.data().rightCols( n2 ) == d2 );
    }

    void test_at( void )
    {
        WLPositions::PositionT p = WLPositions::PositionT::Random();
        WLPositions pos;
        pos.resize( 5 );
        pos.data().col( 3 ) = p;

        TS_ASSERT_EQUALS( pos.at( 3 ), p );
        TS_ASSERT_THROWS( pos.at( pos.size() ), WOutOfBounds );
    }
};

#endif  // WLPOSITIONS_TEST_H_
