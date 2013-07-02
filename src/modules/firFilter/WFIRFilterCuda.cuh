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

#ifndef WFIRFILTERCUDA_CUH_
#define WFIRFILTERCUDA_CUH_

#include <stddef.h>

/**
 * Floating point type (float/double) of a single sample for CUDA kernel. When using double, compute capability >=1.3 is required!
 */
typedef float CuScalarT;

/**
 * Prepare the data and calls the CUDA kernel for FIR filter.
 *
 * \param output A 2 dimensional output array, first dimension = channel, second = samples.
 * \param input A 2 dimensional input array, first dimension = channel, second = samples.
 * \param previous Last samples of previous packet, 2 dimensional.
 * \param channelsNumber of channels.
 * \param samples Number of samples per channel.
 * \param coeffs Coefficient vector.
 * \param coeffSize Coefficient vector size.
 *
 * \return Elapsed time in ms.
 */
extern "C" float cudaFirFilter( CuScalarT* const output,
                const CuScalarT* const input,
                const CuScalarT* const previous,
                size_t channels,
                size_t samples,
                const CuScalarT* const coeffs,
                size_t coeffSize );

#endif  // WFIRFILTERCUDA_CUH_
