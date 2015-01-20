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

#ifndef WLGEOMETRY_TEST_H_
#define WLGEOMETRY_TEST_H_

#include <limits> // double_min

#include <cxxtest/TestSuite.h>

#include <core/common/WLogger.h>

#include "../WLGeometry.h"

/**
 * \author pieloth
 */
class WLGeometryTest: public CxxTest::TestSuite
{
public:
    void setUp( void )
    {
        WLogger::startup();
    }

    void test_isAlmostZero()
    {
        double x;

        x = 0.0;
        TS_ASSERT( WLGeometry::isAlmostZero( x ) );
        x = -1.0 * x;
        TS_ASSERT( WLGeometry::isAlmostZero( x ) );

        x = std::numeric_limits< double >::min() / 2;
        TS_ASSERT( WLGeometry::isAlmostZero( x ) );

        x = std::numeric_limits< double >::min() * 1e6;
        TS_ASSERT( WLGeometry::isAlmostZero( x ) == false );

        x = 0.1;
        TS_ASSERT( WLGeometry::isAlmostZero( x ) == false );

        x = std::numeric_limits< double >::max() / 2;
        TS_ASSERT( WLGeometry::isAlmostZero( x ) == false );
    }

    void test_findOrthogonalVector()
    {
        WLGeometry::Vector3T v;
        WLGeometry::Vector3T o;

        v = WLGeometry::Vector3T::Zero();
        TS_ASSERT( WLGeometry::findOrthogonalVector( &o, v ) == false );

        v = WLGeometry::Vector3T::Random();
        TS_ASSERT( WLGeometry::findOrthogonalVector( &o, v ) );
        TS_ASSERT( WLGeometry::isAlmostZero( v.dot( o ) ) );

        v.setZero();
        v.x() = 8;
        TS_ASSERT( WLGeometry::findOrthogonalVector( &o, v ) );
        TS_ASSERT( WLGeometry::isAlmostZero( v.dot( o ) ) );

        v.setZero();
        v.y() = 23;
        TS_ASSERT( WLGeometry::findOrthogonalVector( &o, v ) );
        TS_ASSERT( WLGeometry::isAlmostZero( v.dot( o ) ) );

        v.setZero();
        v.z() = 42;
        TS_ASSERT( WLGeometry::findOrthogonalVector( &o, v ) );
        TS_ASSERT( WLGeometry::isAlmostZero( v.dot( o ) ) );
    }

    void test_findTagentPlane()
    {
        const double eps = 1e-12;
        WLGeometry::Vector3T n;
        WLGeometry::Vector3T u;
        WLGeometry::Vector3T v;

        n = WLGeometry::Vector3T::Zero();
        TS_ASSERT( WLGeometry::findTagentPlane( &u, &v, n ) == false );

        n = WLGeometry::Vector3T::Random();
        TS_ASSERT( WLGeometry::findTagentPlane( &u, &v, n ) );
        TS_ASSERT_DELTA( n.dot( u ), 0.0, eps );
        TS_ASSERT_DELTA( n.dot( v ), 0.0, eps );
        TS_ASSERT_DELTA( u.dot( v ), 0.0, eps );

        n.setZero();
        n.x() = 42;
        TS_ASSERT( WLGeometry::findTagentPlane( &u, &v, n ) );
        TS_ASSERT_DELTA( n.dot( u ), 0.0, eps );
        TS_ASSERT_DELTA( n.dot( v ), 0.0, eps );
        TS_ASSERT_DELTA( u.dot( v ), 0.0, eps );

        n.setZero();
        n.y() = 23;
        TS_ASSERT( WLGeometry::findTagentPlane( &u, &v, n ) );
        TS_ASSERT_DELTA( n.dot( u ), 0.0, eps );
        TS_ASSERT_DELTA( n.dot( v ), 0.0, eps );
        TS_ASSERT_DELTA( u.dot( v ), 0.0, eps );

        n.setZero();
        n.z() = 8;
        TS_ASSERT( WLGeometry::findTagentPlane( &u, &v, n ) );
        TS_ASSERT_DELTA( n.dot( u ), 0.0, eps );
        TS_ASSERT_DELTA( n.dot( v ), 0.0, eps );
        TS_ASSERT_DELTA( u.dot( v ), 0.0, eps );
    }

    void test_createUpperHalfSphere()
    {
        const double eps = 1e-12;
        const size_t points = 100;
        const float r = 42.0;
        WLGeometry::PointsT sphere;

        TS_ASSERT( WLGeometry::createUpperHalfSphere( &sphere, points, 0.0 ) == 0 );

        TS_ASSERT( WLGeometry::createUpperHalfSphere( &sphere, points, r ) >= points );
        TS_ASSERT( sphere.cols() == points );

        // Test: upper half - 0 <= z <= r
        TS_ASSERT( sphere.row( 2 ).minCoeff() >= 0 );
        TS_ASSERT( sphere.row( 2 ).maxCoeff() <= r );

        // Test: upper half - -r <= x <= r
        TS_ASSERT( -r <= sphere.row( 0 ).minCoeff() && sphere.row( 0 ).minCoeff() < 0.0 );
        TS_ASSERT( 0.0 < sphere.row( 0 ).maxCoeff() && sphere.row( 0 ).maxCoeff() <= r );

        // Test: upper half - -r <= y <= r
        TS_ASSERT( -r <= sphere.row( 1 ).minCoeff() && sphere.row( 1 ).minCoeff() < 0.0 );
        TS_ASSERT( 0.0 < sphere.row( 1 ).maxCoeff() && sphere.row( 1 ).maxCoeff() <= r );

        // Test: radius - for all |p| == r
        TS_ASSERT_DELTA( sphere.colwise().norm().minCoeff(), r, eps );
        TS_ASSERT_DELTA( sphere.colwise().norm().maxCoeff(), r, eps );
    }
};

#endif  // WLGEOMETRY_TEST_H_
