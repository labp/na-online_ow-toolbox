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

#ifndef WHEADPOSITIONCORRECTION_TEST_H_
#define WHEADPOSITIONCORRECTION_TEST_H_

#include <cxxtest/TestSuite.h>
#include <Eigen/Dense>

#include <core/common/WLogger.h>

#include "../WHeadPositionCorrection.h"

/**
 * \author pieloth
 */
class WHeadPositionCorrectionTest: public CxxTest::TestSuite
{
public:
    void setUp( void )
    {
        WLogger::startup();
    }

    void test_generateDipoleSphere( void )
    {
        const double eps = 1e-12;

        WHeadPositionCorrection corr;
        WHeadPositionCorrection::PositionsT pos;
        WHeadPositionCorrection::OrientationsT ori;
        const size_t nDipoles = 300;
        const float r = 0.7;

        TS_ASSERT( corr.generateDipoleSphere( &pos, &ori, nDipoles, r ) );
        TS_ASSERT( pos.cols() >= nDipoles && ori.cols() >= nDipoles );
        TS_ASSERT( pos.cols() == ori.cols() );

        // Test: upper half - 0 <= z <= r
        TS_ASSERT( pos.row( 2 ).minCoeff() >= 0 );
        TS_ASSERT( pos.row( 2 ).maxCoeff() <= r );

        // Test: upper half - -r <= x <= r
        TS_ASSERT( -r <= pos.row( 0 ).minCoeff() && pos.row( 0 ).minCoeff() < 0.0 );
        TS_ASSERT( 0.0 < pos.row( 0 ).maxCoeff() && pos.row( 0 ).maxCoeff() <= r );

        // Test: upper half - -r <= y <= r
        TS_ASSERT( -r <= pos.row( 1 ).minCoeff() && pos.row( 1 ).minCoeff() < 0.0 );
        TS_ASSERT( 0.0 < pos.row( 1 ).maxCoeff() && pos.row( 1 ).maxCoeff() <= r );

        // Test: radius - for all |p| == r
        TS_ASSERT_DELTA( pos.colwise().norm().minCoeff(), r, eps );
        TS_ASSERT_DELTA( pos.colwise().norm().maxCoeff(), r, eps );

        TS_ASSERT( pos.cols() % 2 == 0 ); // 2 Dipoles at each position.

        const size_t offset = pos.cols() / 2;

        for( WHeadPositionCorrection::PositionsT::Index i = 0; i < offset; ++i )
        {
            TS_ASSERT( pos.col( i ) == pos.col( i + offset ) );
            TS_ASSERT_DELTA( pos.col( i ).dot( ori.col( i ) ), 0.0, eps );
            TS_ASSERT_DELTA( pos.col( i ).dot( ori.col( i + offset ) ), 0.0, eps );
            TS_ASSERT_DELTA( ori.col( i ).dot( ori.col( i + offset ) ), 0.0, eps );
        }
    }
};

#endif  // WHEADPOSITIONCORRECTION_TEST_H_
