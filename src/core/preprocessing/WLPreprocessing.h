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

#ifndef WLPREPROCESSING_H_
#define WLPREPROCESSING_H_

#include "core/data/WLDataTypes.h"
#include "core/data/emd/WLEMData.h"
#include "WLWindowFunction.h"

/**
 * Basic preprocessing routines.
 *
 * \author pieloth
 */
namespace WLPreprocessing
{
    typedef WLVector::VectorT VectorT;

    /**
     * Subtracts the baseline from the data.
     *
     * \param dataOut Contains the corrected data. In error case dataOut contains the input data.
     * \param dataIn Data to correct.
     * \param start Start index for mean calculation.
     * \param offset Number of samples for mean calculation.
     *
     * \return True if successful.
     */
    bool baselineCorrection( WLEMData::DataT* const dataOut, const WLEMData::DataT& dataIn, WLSampleIdxT start,
                    WLSampleNrT offset );

    /**
     * Removes linear trend from a vector.
     *
     * \param yptr Corrected output data.
     * \param x Input data.
     */
    void detrend( VectorT* const yptr, const VectorT& x );

    /**
     * Applies a window function on a vector.
     *
     * \param yptr Output data.
     * \param x Input data.
     * \param type Window function to use.
     */
    void windowing( VectorT* const yptr, const VectorT& x, WLWindowFunction::WLEWindow type );
} /* namespace WLPreprocessing */

#endif  // WLPREPROCESSING_H_
