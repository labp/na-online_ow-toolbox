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

#include <Eigen/Core>

#include "WLBoundCalculator.h"

WLBoundCalculator::WLBoundCalculator( WLEMData::ScalarT alpha ) :
                m_alpha( alpha )
{
}

WLBoundCalculator::~WLBoundCalculator()
{
}

WLEMData::ScalarT WLBoundCalculator::getMax( const WLEMData::DataT& data )
{
    const WLEMData::ChannelT::Index channels = data.rows();
    const WLEMData::SampleT::Index samples = data.cols();

    WLEMData::SampleT average( channels );
    average.setZero();

    for( WLEMData::SampleT::Index smp = 0; smp < samples; ++smp )
    {
        average += data.col( smp );
    }
    average *= ( 1.0 / samples );

    WLEMData::ScalarT maxValue = 0;
    for( WLEMData::ChannelT::Index chan = 0; chan < channels; ++chan )
    {
        WLEMData::ScalarT value = 0;
        Eigen::Array< WLEMData::ScalarT, 1, Eigen::Dynamic > vVec( data.row( chan ) );
        vVec -= average( chan );
        vVec *= vVec;
        value = vVec.sum();
        value = m_alpha * sqrt( value / samples ) + average( chan );
        if( value > maxValue )
        {
            maxValue = value;
        }
    }
    return maxValue;
}

WLEMData::ScalarT WLBoundCalculator::getMin( const WLEMData::DataT& data )
{
    return -getMax( data );
}
