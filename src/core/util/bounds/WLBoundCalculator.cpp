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

#include <vector>

#include <boost/assign.hpp>

#include <Eigen/Core>

#include "core/data/emd/WLEMDMEG.h"
#include "core/data/emd/WLEMDSource.h"

#include "WLBoundCalculator.h"

using namespace boost::assign;

WLBoundCalculator::WLBoundCalculator( WLEMData::ScalarT alpha ) :
                m_alpha( alpha )
{
}

WLBoundCalculator::~WLBoundCalculator()
{
}

WLArrayList< WLEMData::ScalarT > WLBoundCalculator::getBounds3D( WLEMMeasurement::ConstSPtr emm, WLEModality::Enum modality )
{
    WLArrayList< WLEMData::ScalarT > bounds;

    if( emm->hasModality( modality ) )
    {
        bounds += getMax( emm->getModality( modality )->getData() );
        bounds += -bounds.at( 0 );

        return bounds;
    }
    if( WLEModality::isMEGCoil( modality ) && emm->hasModality( WLEModality::MEG ) )
    {
        WLEMDMEG::ConstSPtr meg = emm->getModality< const WLEMDMEG >( WLEModality::MEG );
        WLEMDMEG::SPtr megCoil;
        if( WLEMDMEG::extractCoilModality( megCoil, meg, modality, true ) )
        {
            bounds += getMax( megCoil->getData() );
            bounds += -bounds.at( 0 );
            return bounds;
        }
        else
        {
            bounds += 0, 0;
            return bounds;
        }
    }
    else
    {
        bounds += 0, 0;
        return bounds;
    }
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
