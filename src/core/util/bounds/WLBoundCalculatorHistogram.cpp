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

#include <Eigen/Core>

#include <core/common/WLogger.h>

#include "WLBoundCalculatorHistogram.h"

const std::string WLBoundCalculatorHistogram::CLASS = "WLBoundCalculatorHistogram";

WLBoundCalculatorHistogram::WLBoundCalculatorHistogram() :
                m_percent( 80.0 )
{

}

WLEMData::ScalarT WLBoundCalculatorHistogram::getMax( const WLEMData::DataT& data )
{
    return data.cwiseAbs().maxCoeff();
}

WLEMData::ScalarT WLBoundCalculatorHistogram::getMin( const WLEMData::DataT& data )
{
    const int bins = 10; // the number of bins.
    WLEMData::ScalarT minimum = 0;
    const WLEMData::ScalarT p = 1 - ( ( WLEMData::ScalarT )m_percent ) / 100;

    WLEMData::DataT absolute = data.cwiseAbs();

    WLEMData::ScalarT min = absolute.minCoeff(); // calc the absolute minimum.

    absolute = absolute.array() - min; // switch down to 0 as base.
    WLEMData::ScalarT scale = absolute.maxCoeff(); // get the scaling coefficient
    absolute /= scale; // scale into [0;1]

    // divide value range into ten bins
    for( int i = 0; i < bins; ++i )
    {
        // calc the lower limit for each bin.
        WLEMData::ScalarT limit = ( WLEMData::ScalarT )( 1 / ( WLEMData::ScalarT )bins );
        limit *= i;

        if( p <= limit )
        {
            minimum = limit;
            break;
        }
    }

    // scale the minimum to the original data.
    minimum = minimum * ( scale + min );

    return minimum;
}

double WLBoundCalculatorHistogram::getPercent()
{
    return m_percent;
}

void WLBoundCalculatorHistogram::setPercent( double percent )
{
    m_percent = percent;
}
