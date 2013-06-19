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

#include <cstddef>
#include <string>

#include <core/common/WLogger.h>

#include "core/util/WLProfilerLogger.h"
#include "core/util/WLTimeProfiler.h"

#include "WFIRFilter.h"
#include "WFIRFilterCuda.h"
#include "WFIRFilterCuda.cuh"

const std::string WFIRFilterCuda::CLASS = "WFIRFilterCuda";

WFIRFilterCuda::WFIRFilterCuda( WFIRFilter::WEFilterType::Enum filtertype, WFIRFilter::WEWindowsType::Enum windowtype, int order,
                double sFreq, double cFreq1, double cFreq2 ) :
                WFIRFilter( filtertype, windowtype, order, sFreq, cFreq1, cFreq2 )
{
}

WFIRFilterCuda::WFIRFilterCuda( const char *pathToFcf ) :
                WFIRFilter( pathToFcf )
{
}

void WFIRFilterCuda::filter( WLEMData::DataT& out, const WLEMData::DataT& in,
                const WLEMData::DataT& prev )
{
    wlog::debug( CLASS ) << "filter() called!";
    WLTimeProfiler prfTime( CLASS, "filter" );

    const size_t samples = in[0].size();

    SampleT *coeffs = ( SampleT* )malloc( m_coeffitients.size() * sizeof( SampleT ) );
    SampleT *input = ( SampleT* )malloc( in.size() * samples * sizeof( SampleT ) );
    SampleT *previous = ( SampleT* )malloc( prev.size() * m_coeffitients.size() * sizeof( SampleT ) );
    SampleT *output = ( SampleT* )malloc( in.size() * samples * sizeof( SampleT ) );

    // CHANGE from for( size_t i = 0; i < 32; ++i ) to
    for( size_t i = 0; i < in.size(); ++i )
    {
        for( size_t j = 0; j < samples; ++j )
        {
            input[i * samples + j] = ( SampleT )( ( in )[i][j] );
        }
    }

    const size_t prevSamples = m_coeffitients.size();
    for( size_t i = 0; i < prev.size(); ++i )
    {
        for( size_t j = 0; j < prevSamples; ++j )
        {
            previous[i * prevSamples + j] = ( SampleT )( ( prev )[i][j] );
        }
    }

    for( size_t i = 0; i < m_coeffitients.size(); ++i )
    {
        coeffs[i] = ( SampleT )m_coeffitients[i];
    }

    float time = cudaFirFilter( output, input, previous, in.size(), samples, coeffs, m_coeffitients.size() );

    for( size_t i = 0; i < in.size(); ++i )
    {
        WLEMData::ChannelT outChan; // generate a new dimension for every channel
        outChan.reserve( samples );
        // CHANGED to *.assign
//        for( size_t j = 0; j < samples; ++j )
//        {
//            value = ( LaBP::WDataSetEMMEMD::SampleT )output[i * samples + j];
//            outChan.push_back( value );
//        }
        outChan.assign( output + i * samples, output + ( i + 1 ) * samples );
        out.push_back( outChan );
    }

    free( output );
    free( input );
    free( previous );
    free( coeffs );

    WLTimeProfiler prfTimeKernel( CLASS, "filter_kernel", false );
    prfTimeKernel.setMilliseconds( time );
    wlprofiler::log() << prfTimeKernel;
}
