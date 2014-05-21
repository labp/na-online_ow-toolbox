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

#include <cmath>

#include <Eigen/Core>

#include <core/common/WLogger.h>

#include "WLBoundCalculatorHistogram.h"

const std::string WLBoundCalculatorHistogram::CLASS = "WLBoundCalculatorHistogram";

const int WLBoundCalculatorHistogram::DEFAULT_PERCENTAGE = 90;

const int WLBoundCalculatorHistogram::DEFAULT_BINS = 10;

WLBoundCalculatorHistogram::WLBoundCalculatorHistogram() :
                m_percent( DEFAULT_PERCENTAGE ), m_bins( DEFAULT_BINS )
{

}

WLEMData::ScalarT WLBoundCalculatorHistogram::getMax( const WLEMData::DataT& data )
{
    return data.cwiseAbs().maxCoeff();
}

WLEMData::ScalarT WLBoundCalculatorHistogram::getMin( const WLEMData::DataT& data )
{
    if( m_percent == 0.0 )
    {
        return getMax( data );
    }

    WLEMData::ScalarT minimum = 0;
    const WLEMData::ScalarT p = 1 - ( ( WLEMData::ScalarT )m_percent ) / 100;

    WLEMData::DataT absolute = data.cwiseAbs(); // get the data with absolute values

    minimum = absolute.maxCoeff() * p;

    return minimum;
}

double WLBoundCalculatorHistogram::getPercent() const
{
    return m_percent;
}

void WLBoundCalculatorHistogram::setPercent( double percent )
{
    m_percent = percent;
}

int WLBoundCalculatorHistogram::getBins() const
{
    return m_bins;
}

void WLBoundCalculatorHistogram::setBins( int bins )
{
    if( bins == 0 )
    {
        return;
    }

    m_bins = bins;
}
