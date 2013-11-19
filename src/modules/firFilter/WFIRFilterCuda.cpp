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

#include <Eigen/Core>

#include <core/common/WLogger.h>

#include "core/util/profiler/WLProfilerLogger.h"
#include "core/util/profiler/WLTimeProfiler.h"
#include "core/util/WLCudaMacros.h"

#include "WFIRFilter.h"
#include "WFIRFilterCuda.h"
#include "WFIRFilterCuda.cuh"

const std::string WFIRFilterCuda::CLASS = "WFIRFilterCuda";

WFIRFilterCuda::WFIRFilterCuda() : WFIRFilter()
{
}

WFIRFilterCuda::WFIRFilterCuda( WFIRFilter::WEFilterType::Enum filtertype, WFIRFilter::WEWindowsType::Enum windowtype, int order,
                ScalarT sFreq, ScalarT cFreq1, ScalarT cFreq2 ) :
                WFIRFilter( filtertype, windowtype, order, sFreq, cFreq1, cFreq2 )
{
}

WFIRFilterCuda::WFIRFilterCuda( const std::string& pathToFcf ) :
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

    WLEMData::ScalarT *coeffs = ( WLEMData::ScalarT* )malloc( prevSamples * sizeof(WLEMData::ScalarT) );
    WLEMData::ScalarT *input = ( WLEMData::ScalarT* )malloc( channels * samples * sizeof(WLEMData::ScalarT) );
    WLEMData::ScalarT *previous = ( WLEMData::ScalarT* )malloc( channels * prevSamples * sizeof(WLEMData::ScalarT) );
    WLEMData::ScalarT *output = ( WLEMData::ScalarT* )malloc( channels * samples * sizeof(WLEMData::ScalarT) );

    // CHANGE from for( size_t i = 0; i < 32; ++i ) to
    WLEMData::ScalarT* chanWriteTmp;
    for( WLEMData::DataT::Index c = 0; c < channels; ++c )
    {
        chanWriteTmp = input + c * samples;
        for( WLEMData::DataT::Index s = 0; s < samples; ++s )
        {
            chanWriteTmp[s] = ( in( c, s ) );
        }
    }

    for( WLEMData::DataT::Index c = 0; c < channels; ++c )
    {
        chanWriteTmp = previous + c * prevSamples;
        for( WLEMData::DataT::Index s = 0; s < prevSamples; ++s )
        {
            chanWriteTmp[s] = ( prev( c, s ) );
        }
    }

    for( WLEMData::DataT::Index s = 0; s < prevSamples; ++s )
    {
        coeffs[s] = m_coeffitients[s];
    }

    float time = cudaFilter( output, input, previous, channels, samples, coeffs, prevSamples );

    const WLEMData::ScalarT* chanReadTmp;
    for( WLEMData::DataT::Index c = 0; c < in.rows(); ++c )
    {
        chanReadTmp = output + c * samples;
        out.row( c ) = WLEMData::ChannelT::Map( chanReadTmp, samples );
    }

    free( output );
    free( input );
    free( previous );
    free( coeffs );

    WLTimeProfiler prfTimeKernel( CLASS, "filter_kernel", false );
    prfTimeKernel.setMilliseconds( time );
    wlprofiler::log() << prfTimeKernel;
}

float WFIRFilterCuda::cudaFilter( WLEMData::ScalarT* const output, const WLEMData::ScalarT* const input,
                const WLEMData::ScalarT* const previous, size_t channels, size_t samples, const WLEMData::ScalarT* const coeffs,
                size_t coeffSize )
{
    // CudaSafeCall (cudaSetDevice(0));

    CuScalarT *dev_in = NULL;
    size_t pitchIn;

    CuScalarT *dev_prev = NULL;
    size_t pitchPrev;

    CuScalarT *dev_out = NULL;
    size_t pitchOut;

    CuScalarT *dev_co = NULL;

    CudaSafeCall( cudaMallocPitch( &dev_in, &pitchIn, samples * sizeof(CuScalarT), channels ) );
    CudaSafeCall(
                    cudaMemcpy2D( dev_in, pitchIn, input, samples * sizeof(CuScalarT), samples * sizeof(CuScalarT), channels,
                                    cudaMemcpyHostToDevice ) );

    CudaSafeCall( cudaMallocPitch( &dev_prev, &pitchPrev, coeffSize * sizeof(CuScalarT), channels ) );
    CudaSafeCall(
                    cudaMemcpy2D( dev_prev, pitchPrev, previous, coeffSize * sizeof(CuScalarT), coeffSize * sizeof(CuScalarT),
                                    channels, cudaMemcpyHostToDevice ) );

    CudaSafeCall( cudaMallocPitch( &dev_out, &pitchOut, samples * sizeof(CuScalarT), channels ) );

    CudaSafeCall( cudaMalloc( &dev_co, coeffSize * sizeof(CuScalarT) ) );
    CudaSafeCall( cudaMemcpy( dev_co, coeffs, coeffSize * sizeof(CuScalarT), cudaMemcpyHostToDevice ) );

    size_t threadsPerBlock = 32;
    size_t blocksPerGrid = ( samples + threadsPerBlock - 1 ) / threadsPerBlock;
    size_t sharedMem = coeffSize * sizeof(CuScalarT);

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    cudaEventRecord( start, 0 );
    cuFirFilter( blocksPerGrid, threadsPerBlock, sharedMem, dev_out, dev_in, dev_prev, channels, samples, dev_co, coeffSize,
                    pitchOut, pitchIn, pitchPrev );

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

    cudaError_t error = cudaGetLastError();
    if( error != cudaSuccess )
    {
        fprintf( stderr, "cudaFirFilter-Kernel failed: %s\n", cudaGetErrorString( error ) );
    }

    CudaSafeCall(
                    cudaMemcpy2D( output, samples * sizeof(CuScalarT), dev_out, pitchOut, samples * sizeof(CuScalarT), channels,
                                    cudaMemcpyDeviceToHost ) );

    CudaSafeCall( cudaFree( ( void* )dev_in ) );
    CudaSafeCall( cudaFree( ( void* )dev_prev ) );
    CudaSafeCall( cudaFree( ( void* )dev_out ) );
    CudaSafeCall( cudaFree( ( void* )dev_co ) );

    return elapsedTime;
}
