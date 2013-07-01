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

#include <Eigen/Dense>

#include <core/common/WLogger.h>

#include "core/util/profiler/WLProfilerLogger.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WFIRFilter.h"
#include "WFIRFilterCuda.h"
#include "WFIRFilterCuda.cuh"

const std::string WFIRFilterCuda::CLASS = "WFIRFilterCuda";

WFIRFilterCuda::WFIRFilterCuda( WFIRFilter::WEFilterType::Enum filtertype, WFIRFilter::WEWindowsType::Enum windowtype, int order,
                ScalarT sFreq, ScalarT cFreq1, ScalarT cFreq2 ) :
                WFIRFilter( filtertype, windowtype, order, sFreq, cFreq1, cFreq2 )
{
}

WFIRFilterCuda::WFIRFilterCuda( const char *pathToFcf ) :
                WFIRFilter( pathToFcf )
{
}

void WFIRFilterCuda::filter( WLEMData::DataT& out, const WLEMData::DataT& in, const WLEMData::DataT& prev )
{
    wlog::debug( CLASS ) << "filter() called!";
    WLTimeProfiler prfTime( CLASS, "filter" );

    const WLEMData::DataT::Index channels = in.rows();
    const WLEMData::DataT::Index samples = in.cols();
    const WLEMData::DataT::Index prevSamples = static_cast< WLEMData::DataT::Index >( m_coeffitients.size() );

    SampleT *coeffs = ( SampleT* )malloc( prevSamples * sizeof(SampleT) );
    SampleT *input = ( SampleT* )malloc( channels * samples * sizeof(SampleT) );
    SampleT *previous = ( SampleT* )malloc( channels * prevSamples * sizeof(SampleT) );
    SampleT *output = ( SampleT* )malloc( channels * samples * sizeof(SampleT) );

    // CHANGE from for( size_t i = 0; i < 32; ++i ) to
    SampleT* chanWriteTmp;
    for( WLEMData::DataT::Index c = 0; c < channels; ++c )
    {
        chanWriteTmp = input + c * samples;
        for( WLEMData::DataT::Index s = 0; s < samples; ++s )
        {
            chanWriteTmp[s] = ( SampleT )( in( c, s ) );
        }
    }

    for( WLEMData::DataT::Index c = 0; c < channels; ++c )
    {
        chanWriteTmp = previous + c * prevSamples;
        for( WLEMData::DataT::Index s = 0; s < prevSamples; ++s )
        {
            chanWriteTmp[s] = ( SampleT )( prev( c, s ) );
        }
    }

    for( WLEMData::DataT::Index s = 0; s < prevSamples; ++s )
    {
        coeffs[s] = ( SampleT )m_coeffitients[s];
    }

    float time = cudaFirFilter( output, input, previous, channels, samples, coeffs, prevSamples );

    const SampleT* chanReadTmp;
    for( WLEMData::DataT::Index c = 0; c < in.rows(); ++c )
    {
        chanReadTmp = output + c * samples;
        out.row( c ) = Eigen::VectorXf::Map( chanReadTmp, samples ).cast< WLEMData::SampleT >();
    }

    free( output );
    free( input );
    free( previous );
    free( coeffs );

    WLTimeProfiler prfTimeKernel( CLASS, "filter_kernel", false );
    prfTimeKernel.setMilliseconds( time );
    wlprofiler::log() << prfTimeKernel;
}
