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

#ifndef WLCUDAMACROS_H_
#define WLCUDAMACROS_H_

/**
 * \ingroup util
 */

#ifdef FOUND_CUDA

#include <sstream>
#include <string>

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <core/common/WLogger.h>
#include <core/common/WException.h>

#include "core/exception/WLBadAllocException.h"

#define CudaSafeCall( err )     __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError_t err, const char *file, const int line )
{
    if( err != cudaSuccess )
    {
        wlog::error( "CudaSafeCall" ) << "Failed at " << file << ":" << line << " " << cudaGetErrorString( err );
    }
}

#define CudaThrowsCall( err )     __cudaThrowsCall( err, __FILE__, __LINE__ )

inline void __cudaThrowsCall( cudaError_t err, const char *file, const int line )
{
    if( err != cudaSuccess )
    {
        std::stringstream sstream;
        sstream << "Failed at " << file << ":" << line << " " << cudaGetErrorString( err );
        const std::string msg( sstream.str() );
        switch( err )
        {
            case cudaErrorMemoryAllocation:
                throw WLBadAllocException( msg );
            default:
                throw WException( msg );
        }
    }
}

#endif  // FOUND_CUDA

#endif  // WLCUDAMACROS_H_
