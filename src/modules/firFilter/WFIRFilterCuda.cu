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

#include "core/util/WLCudaMacros.h"

#include "WFIRFilterCuda.cuh"

// -----------
// CUDA kernel
// -----------
// Old implementation (slow):
// One thread per channel has no coalesced memory access and a bad scalability for fewer channels than stream processors.
// This implementation (faster):
// One thread calculates one element of a channel.
// All threads are working on the same channel at the same time.

__global__ void cu_fir_filter_s( float* const out, const float* const in, const float* const prev, size_t channels,
                size_t samples, const float* const coeff, size_t coeffSize, size_t pitchOut, size_t pitchIn, size_t pitchPrev )
{
    typedef float CuScalarT;
    // Collect coefficients to shared memory
    extern __shared__ CuScalarT coeffSharedS[];
    for( size_t i = threadIdx.x; i < coeffSize; i += blockDim.x )
    {
        coeffSharedS[i] = coeff[i];
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
    pitchIn = pitchIn / sizeof(CuScalarT);
    pitchPrev = pitchPrev / sizeof(CuScalarT);
    pitchOut = pitchOut / sizeof(CuScalarT);

    for( size_t chan = 0; chan < channels; ++chan )
    {
        for( size_t k = 0; k < coeffSize; ++k )
        {
            // CHANGED tmp += ( gid >= k ) ? coeffShared[k] * rowIn[gid - k] : 0;
            if( gid >= k )
            {
                tmp += coeffSharedS[k] * rowIn[gid - k];
            }
            else
            {
                tmp += coeffSharedS[k] * rowPrev[coeffSize - ( k - gid )];
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

__global__ void cu_fir_filter_d( double* const out, const double* const in, const double* const prev, size_t channels,
                size_t samples, const double* const coeff, size_t coeffSize, size_t pitchOut, size_t pitchIn, size_t pitchPrev )
{
    typedef double CuScalarT;
    // Collect coefficients to shared memory
    extern __shared__ CuScalarT coeffSharedD[];
    for( size_t i = threadIdx.x; i < coeffSize; i += blockDim.x )
    {
        coeffSharedD[i] = coeff[i];
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
    pitchIn = pitchIn / sizeof(CuScalarT);
    pitchPrev = pitchPrev / sizeof(CuScalarT);
    pitchOut = pitchOut / sizeof(CuScalarT);

    for( size_t chan = 0; chan < channels; ++chan )
    {
        for( size_t k = 0; k < coeffSize; ++k )
        {
            // CHANGED tmp += ( gid >= k ) ? coeffShared[k] * rowIn[gid - k] : 0;
            if( gid >= k )
            {
                tmp += coeffSharedD[k] * rowIn[gid - k];
            }
            else
            {
                tmp += coeffSharedD[k] * rowPrev[coeffSize - ( k - gid )];
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

// --------------------
// Wrapper kernel calls
// --------------------

void cuFirFilter( const size_t GRID, const size_t BLOCK, const size_t SHARED, float* const output, const float* const input,
                const float* const previous, size_t channels, size_t samples, const float* const coeffs, size_t coeffSize,
                size_t pitchOut, size_t pitchIn, size_t pitchPrev )
{
    cu_fir_filter_s CUDA_KERNEL_DIM( GRID, BLOCK, SHARED ) ( output, input, previous, channels, samples, coeffs, coeffSize, pitchOut, pitchIn, pitchPrev );
}

void cuFirFilter( const size_t GRID, const size_t BLOCK, const size_t SHARED, double* const output, const double* const input,
                const double* const previous, size_t channels, size_t samples, const double* const coeffs, size_t coeffSize,
                size_t pitchOut, size_t pitchIn, size_t pitchPrev )
{
    cu_fir_filter_d CUDA_KERNEL_DIM( GRID, BLOCK, SHARED ) ( output, input, previous, channels, samples, coeffs, coeffSize, pitchOut, pitchIn, pitchPrev );
}
