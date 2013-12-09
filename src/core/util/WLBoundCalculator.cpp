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

#include <Eigen/Core>

#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMEnumTypes.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDMEG.h"
#include "core/data/emd/WLEMDSource.h"

#include "WLBoundCalculator.h"

namespace LaBP
{
    WLBoundCalculator::WLBoundCalculator( WLEMData::ScalarT alpha ) :
                    m_alpha( alpha )
    {
    }

    WLBoundCalculator::~WLBoundCalculator()
    {
    }

    WLEMData::ScalarT WLBoundCalculator::getMax2D( WLEMMeasurement::ConstSPtr emm, LaBP::WEModalityType::Enum modality )
    {
        if( modality == LaBP::WEModalityType::SOURCE && emm->hasModality( modality ) )
        {
            LaBP::WEModalityType::Enum origin_modality =
                            emm->getModality< const WLEMDSource >( modality )->getOriginModalityType();

            return getMax( emm->getModality( origin_modality )->getData() );
        }
        if( WLEMDMEG::isMegType( modality ) )
        {
            modality = WEModalityType::MEG;
        }
        if( emm->hasModality( modality ) )
        {
            return getMax( emm->getModality( modality )->getData() );
        }
        else
        {
            return 0;
        }
    }

    WLEMData::ScalarT WLBoundCalculator::getMax3D( WLEMMeasurement::ConstSPtr emm, LaBP::WEModalityType::Enum modality )
    {
        if( emm->hasModality( modality ) )
        {
            return getMax( emm->getModality( modality )->getData() );
        }
        if( WLEMDMEG::isMegType( modality ) && modality != WEModalityType::MEG && emm->hasModality( WEModalityType::MEG ) )
        {
            WLEMDMEG::ConstSPtr meg = emm->getModality< const WLEMDMEG >( WEModalityType::MEG );
            WLEMDMEG::SPtr megCoil;
            if( WLEMDMEG::extractCoilModality( megCoil, meg, modality, true ) )
            {
                return getMax( megCoil->getData() );
            }
            else
            {
                return 0;
            }
        }
        else
        {
            return 0;
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

} /* namespace LaBP */
