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

#include <cstddef>
#include <string>

#include <Eigen/Core>

#include <core/common/WLogger.h>
#include <core/common/WException.h>

#include "core/exception/WLBadAllocException.h"
#include "core/util/profiler/WLProfilerLogger.h"
#include "core/util/profiler/WLTimeProfiler.h"
#include "core/util/WLCudaMacros.h"
#include "core/util/WLCudaMacrosNVCC.h"

#include "WFIRFilter.h"
#include "WFIRFilterCuda.h"
#include "WFIRFilterCuda.cuh"

const std::string WFIRFilterCuda::CLASS = "WFIRFilterCuda";

WFIRFilterCuda::WFIRFilterCuda() :
                WFIRFilter()
{
}

WFIRFilterCuda::WFIRFilterCuda( WFIRFilter::WEFilterType::Enum filtertype, WLWindowFunction::WLEWindow windowtype, int order,
                WLFreqT sFreq, WLFreqT cFreq1, WLFreqT cFreq2 ) :
                WFIRFilter( filtertype, windowtype, order, sFreq, cFreq1, cFreq2 )
{
}

WFIRFilterCuda::WFIRFilterCuda( const std::string& pathToFcf ) :
                WFIRFilter( pathToFcf )
{
}

WFIRFilterCuda::~WFIRFilterCuda()
{
}

bool WFIRFilterCuda::filter( WLEMData::DataT& out, const WLEMData::DataT& in, const WLEMData::DataT& prev )
{
    wlog::debug( CLASS ) << "filter() called!";
    WLTimeProfiler prfTime( CLASS, "filter" );
    bool rc = true;

    const WLEMData::DataT::Index channels = in.rows();
    const WLEMData::DataT::Index samples = in.cols();
    const WLEMData::DataT::Index prevSamples = static_cast< WLEMData::DataT::Index >( m_coeffitients.size() );

    WLEMData::ScalarT *coeffs = ( WLEMData::ScalarT* )malloc( prevSamples * sizeof( WLEMData::ScalarT ) );
    WLEMData::ScalarT *input = ( WLEMData::ScalarT* )malloc( channels * samples * sizeof( WLEMData::ScalarT ) );
    WLEMData::ScalarT *previous = ( WLEMData::ScalarT* )malloc( channels * prevSamples * sizeof( WLEMData::ScalarT ) );
    WLEMData::ScalarT *output = ( WLEMData::ScalarT* )malloc( channels * samples * sizeof( WLEMData::ScalarT ) );

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

    try
    {
        float time = cudaFilter( output, input, previous, channels, samples, coeffs, prevSamples );
        WLTimeProfiler prfTimeKernel( CLASS, "filter_kernel", false );
        prfTimeKernel.setMilliseconds( time );
        wlprofiler::log() << prfTimeKernel;

        const WLEMData::ScalarT* chanReadTmp;
        for( WLEMData::DataT::Index c = 0; c < in.rows(); ++c )
        {
            chanReadTmp = output + c * samples;
            out.row( c ) = WLEMData::ChannelT::Map( chanReadTmp, samples );
        }
    }
    catch( const WException& e )
    {
        wlog::error( CLASS ) << e.what();
        rc = false;
    }

    free( output );
    free( input );
    free( previous );
    free( coeffs );

    return rc;
}

float WFIRFilterCuda::cudaFilter( WLEMData::ScalarT* const output, const WLEMData::ScalarT* const input,
                const WLEMData::ScalarT* const previous, size_t channels, size_t samples, const WLEMData::ScalarT* const coeffs,
                size_t coeffSize )
{
    CuScalarT *dev_in = NULL;
    size_t pitchIn;

    CuScalarT *dev_prev = NULL;
    size_t pitchPrev;

    CuScalarT *dev_out = NULL;
    size_t pitchOut;

    CuScalarT *dev_co = NULL;

    try
    {
        CudaThrowsCall( cudaMallocPitch( ( void** )&dev_in, &pitchIn, samples * sizeof( CuScalarT ), channels ) );
        CudaThrowsCall(
                        cudaMemcpy2D( dev_in, pitchIn, input, samples * sizeof( CuScalarT ), samples * sizeof( CuScalarT ),
                                        channels, cudaMemcpyHostToDevice ) );

        CudaThrowsCall( cudaMallocPitch( ( void** )&dev_prev, &pitchPrev, coeffSize * sizeof( CuScalarT ), channels ) );
        CudaThrowsCall(
                        cudaMemcpy2D( dev_prev, pitchPrev, previous, coeffSize * sizeof( CuScalarT ),
                                        coeffSize * sizeof( CuScalarT ), channels, cudaMemcpyHostToDevice ) );

        CudaThrowsCall( cudaMallocPitch( ( void** )&dev_out, &pitchOut, samples * sizeof( CuScalarT ), channels ) );

        CudaThrowsCall( cudaMalloc( ( void** )&dev_co, coeffSize * sizeof( CuScalarT ) ) );
        CudaThrowsCall( cudaMemcpy( dev_co, coeffs, coeffSize * sizeof( CuScalarT ), cudaMemcpyHostToDevice ) );
    }
    catch( const WException& e )
    {
        wlog::error( CLASS ) << e.what();
        if( dev_in )
        {
            CudaSafeCall( cudaFree( ( void* )dev_in ) );
        }
        if( dev_prev )
        {
            CudaSafeCall( cudaFree( ( void* )dev_prev ) );
        }
        if( dev_out )
        {
            CudaSafeCall( cudaFree( ( void* )dev_out ) );
        }
        if( dev_co )
        {
            CudaSafeCall( cudaFree( ( void* )dev_co ) );
        }
        throw WLBadAllocException( "Could not allocate CUDA memory!" );
    }

    size_t threadsPerBlock = 32;
    size_t blocksPerGrid = ( samples + threadsPerBlock - 1 ) / threadsPerBlock;
    size_t sharedMem = coeffSize * sizeof( CuScalarT );

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    cudaEventRecord( start, 0 );
    cuFirFilter( blocksPerGrid, threadsPerBlock, sharedMem, dev_out, dev_in, dev_prev, channels, samples, dev_co, coeffSize,
                    pitchOut, pitchIn, pitchPrev );
    cudaError_t kernelError = cudaGetLastError();

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    try
    {
        if( kernelError != cudaSuccess )
        {
            const std::string err( cudaGetErrorString( kernelError ) );
            throw WException( "CUDA kernel failed: " + err );
        }
        CudaThrowsCall(
                        cudaMemcpy2D( output, samples * sizeof( CuScalarT ), dev_out, pitchOut, samples * sizeof( CuScalarT ),
                                        channels, cudaMemcpyDeviceToHost ) );
    }
    catch( const WException& e )
    {
        wlog::error( CLASS ) << e.what();
        elapsedTime = -1.0;
    }

    CudaSafeCall( cudaFree( ( void* )dev_in ) );
    CudaSafeCall( cudaFree( ( void* )dev_prev ) );
    CudaSafeCall( cudaFree( ( void* )dev_out ) );
    CudaSafeCall( cudaFree( ( void* )dev_co ) );

    if( elapsedTime > -1.0 )
    {
        return elapsedTime;
    }
    else
    {
        throw WException( "Error in cudaFilter()" );
    }
}
