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

#include "core/data/WLMatrixTypes.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMEnumTypes.h"
#include "core/data/emd/WLEMD.h"
#include "core/data/emd/WLEMDSource.h"

#include "WLBoundCalculator.h"

namespace LaBP
{
    WLBoundCalculator::WLBoundCalculator( LaBP::WLEMD::SampleT alpha ) :
                    m_alpha( alpha )
    {
    }

    WLBoundCalculator::~WLBoundCalculator()
    {
    }

    LaBP::WLEMD::SampleT WLBoundCalculator::getMax2D( WLEMMeasurement::ConstSPtr emm,
                    LaBP::WEModalityType::Enum modality )
    {
        if( modality == LaBP::WEModalityType::SOURCE )
        {
            LaBP::WEModalityType::Enum origin_modality =
                            emm->getModality< const LaBP::WLEMDSource >( modality )->getOriginModalityType();

            return getMax( emm->getModality( origin_modality )->getData() );
        }
        else
        {
            return getMax( emm->getModality( modality )->getData() );
        }
    }

    LaBP::WLEMD::SampleT WLBoundCalculator::getMax3D( WLEMMeasurement::ConstSPtr emm,
                    LaBP::WEModalityType::Enum modality )
    {
        if( modality == LaBP::WEModalityType::SOURCE )
        {
            return getMax( emm->getModality< const LaBP::WLEMDSource >( modality )->getMatrix() );
        }
        else
        {
            return getMax( emm->getModality( modality )->getData() );
        }
    }

    LaBP::WLEMD::SampleT WLBoundCalculator::getMax( const MatrixT& matrix )
    {
        std::vector< LaBP::WLEMD::SampleT > average;
        for( MatrixT::Index r = 0; r < matrix.rows(); ++r )
        {
            LaBP::WLEMD::SampleT sum = 0;
            for( MatrixT::Index c = 0; c < matrix.cols(); ++c )
            {
                sum += matrix( r, c );
            }
            average.push_back( sum / matrix.cols() );
        }

        LaBP::WLEMD::SampleT maxValue = 0;
        for( MatrixT::Index r = 0; r < matrix.rows(); ++r )
        {
            LaBP::WLEMD::SampleT value = 0;
            for( MatrixT::Index c = 0; c < matrix.cols(); ++c )
            {
                value += ( matrix( r, c ) - average[r] ) * ( matrix( r, c ) - average[r] );
            }
            value = sqrt( value / matrix.cols() ) * m_alpha + average[r];
            if( value > maxValue )
            {
                maxValue = value;
            }
        }
        return maxValue;
    }

    LaBP::WLEMD::SampleT WLBoundCalculator::getMax( const LaBP::WLEMD::DataT& data )
    {
        std::vector< LaBP::WLEMD::SampleT > average;
        const size_t channels = data.size();
        for( size_t chan = 0; chan < channels; ++chan )
        {
            LaBP::WLEMD::SampleT sum = 0;
            const size_t samples = data[chan].size();
            for( size_t smp = 0; smp < samples; ++smp )
            {
                sum += data[chan][smp];
            }
            average.push_back( sum / samples );
        }

        LaBP::WLEMD::SampleT maxValue = 0;
        for( size_t chan = 0; chan < channels; ++chan )
        {
            LaBP::WLEMD::SampleT value = 0;
            const size_t samples = data[chan].size();
            for( size_t smp = 0; smp < samples; ++smp )
            {
                value += ( data[chan][smp] - average[chan] ) * ( data[chan][smp] - average[chan] );
            }
            value = sqrt( value / samples ) * m_alpha + average[chan];
            if( value > maxValue )
            {
                maxValue = value;
            }
        }
        return maxValue;
    }

} /* namespace LaBP */
