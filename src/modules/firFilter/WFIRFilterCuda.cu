//---------------------------------------------------------------------------
//
// Project: OpenWalnut ( http://www.openwalnut.org )
//
// Copyright 2009 OpenWalnut Community, BSV@Uni-Leipzig and CNCF@MPI-CBS, Copyright 2010 RRZK University of Cologne
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

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "WFIRFilterCuda.cuh"

// to prevent IDE complains about unknown cuda-keywords
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __host__
#define __shared__
#define __syncthreads();
#define CUDA_KERNEL_DIM( ... )

#else
#define CUDA_KERNEL_DIM( ... )  <<< __VA_ARGS__ >>>

#endif

#define CudaSafeCall( err )     __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError_t err, const char *file, const int line )
{
    if( err != cudaSuccess )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
    }
}

// Old implementation (slow):
// One thread per channel has no coalesced memory access and a bad scalability for fewer channels than stream processors.
// This implementation (faster):
// One thread calculates one element of a channel.
// All threads are working on the same channel at the same time.
__global__ void dev_cudaFirFilter( CuScalarT* const out, const CuScalarT* const in, const CuScalarT* const prev, size_t channels,
                size_t samples, const CuScalarT* const coeff, size_t coeffSize, size_t pitchOut, size_t pitchIn, size_t pitchPrev )
{
    // Collect coefficients to shared memory
    extern __shared__ CuScalarT coeffShared[];
    for( size_t i = threadIdx.x; i < coeffSize; i += blockDim.x )
    {
        coeffShared[i] = coeff[i];
    }
    // Wait until all coefficients are stored
    __syncthreads();

    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if( gid >= samples )
        return;

    // Init values for first channel
    CuScalarT tmp = 0.0;
    const CuScalarT* rowIn = in;
    const CuScalarT* rowPrev = prev;
    CuScalarT* rowOut = out;

    // Convert pitch in bytes to pitch in elements (width)
    pitchIn = pitchIn / sizeof( CuScalarT );
    pitchPrev = pitchPrev / sizeof( CuScalarT );
    pitchOut = pitchOut / sizeof( CuScalarT );

    for( size_t chan = 0; chan < channels; ++chan )
    {
        for( size_t k = 0; k < coeffSize; ++k )
        {
            // CHANGED tmp += ( gid >= k ) ? coeffShared[k] * rowIn[gid - k] : 0;
            if( gid >= k )
            {
                tmp += coeffShared[k] * rowIn[gid - k];
            }
            else
            {
                tmp += coeffShared[k] * rowPrev[coeffSize - ( k - gid )];
            }
        }
        rowOut[gid] = tmp;

        // Prepare next channel
        tmp = 0.0;
        rowIn += pitchIn;
        rowPrev += pitchPrev;
        rowOut += pitchOut;
    }
}

float cudaFirFilter( CuScalarT* const output, const CuScalarT* const input, const CuScalarT* const previous, size_t channels,
                size_t samples, const CuScalarT* const coeffs, size_t coeffSize )
{
    // CudaSafeCall (cudaSetDevice(0));

    CuScalarT *dev_in = NULL;
    size_t pitchIn;

    CuScalarT *dev_prev = NULL;
    size_t pitchPrev;

    CuScalarT *dev_out = NULL;
    size_t pitchOut;

    CuScalarT *dev_co = NULL;

    CudaSafeCall( cudaMallocPitch( &dev_in, &pitchIn, samples * sizeof( CuScalarT ), channels ) );
    CudaSafeCall(
                    cudaMemcpy2D( dev_in, pitchIn, input, samples * sizeof( CuScalarT ), samples * sizeof( CuScalarT ), channels, cudaMemcpyHostToDevice ) );

    CudaSafeCall( cudaMallocPitch( &dev_prev, &pitchPrev, coeffSize * sizeof( CuScalarT ), channels ) );
    CudaSafeCall(
                    cudaMemcpy2D( dev_prev, pitchPrev, previous, coeffSize * sizeof( CuScalarT ), coeffSize * sizeof( CuScalarT ), channels, cudaMemcpyHostToDevice ) );

    CudaSafeCall( cudaMallocPitch( &dev_out, &pitchOut, samples * sizeof( CuScalarT ), channels ) );

    CudaSafeCall( cudaMalloc( &dev_co, coeffSize * sizeof( CuScalarT ) ) );
    CudaSafeCall( cudaMemcpy( dev_co, coeffs, coeffSize * sizeof( CuScalarT ), cudaMemcpyHostToDevice ) );

    size_t threadsPerBlock = 32;
    size_t blocksPerGrid = ( samples + threadsPerBlock - 1 ) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    cudaEventRecord( start, 0 );
    dev_cudaFirFilter CUDA_KERNEL_DIM( blocksPerGrid, threadsPerBlock, coeffSize * sizeof( CuScalarT ) ) ( dev_out, dev_in, dev_prev, channels, samples, dev_co, coeffSize, pitchOut, pitchIn, pitchPrev );

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
                    cudaMemcpy2D( output, samples * sizeof( CuScalarT ), dev_out, pitchOut, samples * sizeof( CuScalarT ), channels, cudaMemcpyDeviceToHost ) );

    CudaSafeCall( cudaFree( ( void* )dev_in ) );
    CudaSafeCall( cudaFree( ( void* )dev_prev ) );
    CudaSafeCall( cudaFree( ( void* )dev_out ) );
    CudaSafeCall( cudaFree( ( void* )dev_co ) );

    return elapsedTime;
}
