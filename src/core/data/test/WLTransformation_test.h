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

#ifndef WLTRANSFORMATION_TEST_H_
#define WLTRANSFORMATION_TEST_H_

#include <cxxtest/TestSuite.h>
#include <Eigen/Dense>  // homogeneous

#include <core/common/WLogger.h>
#include <core/common/exceptions/WPreconditionNotMet.h>

#include "../WLPositions.h"
#include "../WLTransformation.h"

class WLTransformationTest: public CxxTest::TestSuite
{
public:
    void setUp( void )
    {
        WLogger::startup();
    }

    void test_rotation( void )
    {
        WLTransformation t;
        t.data() = WLTransformation::TransformationT::Random();
        WLTransformation::RotationT r = t.rotation();
        TS_ASSERT_EQUALS( r.rows(), 3 );
        TS_ASSERT_EQUALS( r.cols(), 3 );
        TS_ASSERT( r == t.data().topLeftCorner( 3, 3 ) );
    }

    void test_translation( void )
    {
        WLTransformation t;
        t.data() = WLTransformation::TransformationT::Random();
        WLTransformation::TranslationT trl = t.translation();
        TS_ASSERT_EQUALS( trl.rows(), 3 );
        TS_ASSERT_EQUALS( trl.cols(), 1 );
        TS_ASSERT( trl == t.data().topRightCorner( 3, 1 ) );
    }

    void test_transformFailCoordSystem()
    {
        WLPositions pos;
        pos.resize( 3 );
        pos.coordSystem( WLECoordSystem::HEAD );
        WLTransformation t;
        t.from( WLECoordSystem::DEVICE );
        TS_ASSERT_THROWS( t * pos, WPreconditionNotMet );
    }

    void test_transformFailUnit()
    {
        WLPositions pos;
        pos.resize( 3 );
        pos.unit( WLEUnit::METER );
        WLTransformation t;
        t.unit( WLEUnit::VOLT );
        TS_ASSERT_THROWS( t * pos, WPreconditionNotMet );
    }

    void test_transformFailExponent()
    {
        WLPositions pos;
        pos.resize( 3 );
        pos.unit( WLEUnit::METER );
        pos.exponent( WLEExponent::CENTI );
        WLTransformation t;
        t.unit( WLEUnit::METER );
        t.exponent( WLEExponent::MILLI );
        TS_ASSERT_THROWS( t * pos, WPreconditionNotMet );
    }

    void test_transformOk()
    {
        const WLPositions::IndexT n = 5;
        const WLPositions::PositionsT d1 = WLPositions::PositionsT::Random( 3, n );
        WLPositions pos;
        pos.resize( 3 );
        pos.unit( WLEUnit::METER );
        pos.exponent( WLEExponent::BASE );
        pos.data( d1 );

        WLTransformation t;
        t.unit( WLEUnit::METER );
        t.exponent( WLEExponent::BASE );
        t.data( WLTransformation::TransformationT::Identity() );

        WLPositions::SPtr res = t * pos;
        TS_ASSERT_EQUALS( res->size(), n );
        TS_ASSERT( res->data() == d1 );

        const WLTransformation::TranslationT trl( 3.0, 4.0, 5.0 );
        const WLPositions::PositionsT d2 = d1.colwise() + trl;
        t.data().block( 0, 3, 3, 1 ) = trl;
        res = t * pos;
        TS_ASSERT_EQUALS( res->size(), n );
        TS_ASSERT( res->data() == d2 );

        WLTransformation::TransformationT trans = WLTransformation::TransformationT::Identity();
        trans.block( 0, 3, 3, 1 ) = trl;
        trans.block( 0, 0, 3, 3 ) = WLTransformation::RotationT::Random();
        const WLPositions::PositionsT d3 = ( trans * d1.colwise().homogeneous() ).block( 0, 0, 3, n );

        t.data( trans );
        res = t * pos;
        TS_ASSERT_EQUALS( res->size(), n );
        TS_ASSERT( res->data() == d3 );
    }

    void test_inverse()
    {
        WLECoordSystem::Enum from = WLECoordSystem::DEVICE;
        WLECoordSystem::Enum to = WLECoordSystem::HEAD;
        WLEUnit::Enum unit = WLEUnit::METER;
        WLEExponent::Enum exp = WLEExponent::MICRO;
        WLTransformation::TransformationT dat = WLTransformation::TransformationT::Random();
        dat( 3, 3 ) = 1;

        WLTransformation t;
        t.from( from );
        t.to( to );
        t.unit( unit );
        t.exponent( exp );
        t.data( dat );

        WLTransformation::SPtr ti = t.inverse();

        TS_ASSERT_EQUALS( ti->from(), to );
        TS_ASSERT_EQUALS( ti->to(), from );
        TS_ASSERT_EQUALS( ti->unit(), unit );
        TS_ASSERT_EQUALS( ti->exponent(), exp );
        TS_ASSERT_EQUALS( ti->data(), dat.inverse() );
    }
};

#endif  // WLTRANSFORMATION_TEST_H_
