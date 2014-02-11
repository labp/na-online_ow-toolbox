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

#ifndef WLCUDAMACROSNVCC_H_
#define WLCUDAMACROSNVCC_H_

#ifdef LABP_FLOAT_COMPUTATION
typedef float CuScalarT;
#else
typedef double CuScalarT;
#endif  // LABP_FLOAT_COMPUTATION

#ifdef FOUND_CUDA

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

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

#define CuSafeCall( err )     __CuSafeCall( err, __FILE__, __LINE__ )

inline void __CuSafeCall( cudaError_t err, const char *file, const int line )
{
    if( err != cudaSuccess )
    {
        fprintf( stderr, "CuSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
    }
}

#endif  // FOUND_CUDA

#endif  // WLCUDAMACROSNVCC_H_
