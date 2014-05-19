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
    WLEMData::ScalarT max = getMax( data );

    return max * ( 1 - m_percent / 100 );
}

double WLBoundCalculatorHistogram::getPercent()
{
    return m_percent;
}

void WLBoundCalculatorHistogram::setPercent( double percent )
{
    m_percent = percent;
}
