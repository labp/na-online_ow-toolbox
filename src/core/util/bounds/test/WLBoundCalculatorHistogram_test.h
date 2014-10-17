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

#ifndef WLBOUNDCALCULATORHISTOGRAM_TEST_H_
#define WLBOUNDCALCULATORHISTOGRAM_TEST_H_

#include <boost/assign.hpp>

#include <Eigen/Core>

#include <cxxtest/TestSuite.h>

#include <core/common/WLogger.h>

#include "core/util/bounds/WLBoundCalculatorHistogram.h"

using namespace boost::assign;

class WLBoundCalculatorHistogramTest: public CxxTest::TestSuite
{
public:

    void setUp( void )
    {
        WLogger::startup();
    }

    void test_boundCalc()
    {
        TS_TRACE( "Test Histogram Boundaries Calcluation." );

        WLBoundCalculatorHistogram::SPtr calc( new WLBoundCalculatorHistogram );

        double p = 90; // the lower border

        // define the test matrix
        Eigen::MatrixXd A( 4, 5 );
        A << 1.0, 0.2, -0.8, 1.5, -1.1, 0.7, 0.2, -0.6, 1.1, -1.5, 0.8, 0.1, -1.8, 1.0, -1.0, 1.0, 0.5, -1.5, 1.4, -0.7;

        calc->setPercent( p );

        WLEMData::ScalarT max = calc->getMax( A );
        TS_ASSERT( max == 1.8 ); // test the absolute maximum
        TS_TRACE( "Maximum determined:" );
        TS_TRACE( max );

        WLEMData::ScalarT min = calc->getMin( A );
        TS_ASSERT_EQUALS( ( ( int )( min * 100 ) ), 18 );
        TS_TRACE( "Minimum determined:" );
        TS_TRACE( min );
    }
};

#endif  // WLBOUNDCALCULATORHISTOGRAM_TEST_H_
