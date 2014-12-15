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

#ifndef WLWINDOWFUNCTION_H_
#define WLWINDOWFUNCTION_H_

#include <set>
#include <string>

#include "core/data/WLDataTypes.h"

/**
 * \brief Window functions.
 *
 * Window functions, e.g. for FIR filter.
 *
 * \author pieloth
 * \ingroup preproc
 */
namespace WLWindowFunction
{
    typedef WLVector::VectorT VectorT;

    enum WLEWindow
    {
        HAMMING, RECTANGLE, BARLETT, BLACKMAN, HANNING, UNKNOWN
    };

    /**
     * Gets a set of all supported window functions.
     *
     * \return A set of all supported window functions.
     */
    std::set< WLEWindow > values();

    /**
     * Gets the name of a window function.
     *
     * \param value identifier
     * \return Name for the identifier.
     */
    std::string name( WLEWindow value );

    /**
     *  Calculates the factors for a window function.
     *
     * \param samples Number of samples.
     * \param type Window function to use.
     *
     * \return A vector of factors.
     */
    VectorT window( WLSampleNrT samples, WLEWindow type );

    VectorT hamming( WLSampleNrT samples ); /**< Hamming window function. */

    VectorT rectangle( WLSampleNrT samples ); /**< Rectangle window. */

    VectorT barlett( WLSampleNrT samples ); /**< Barlett window function. */

    VectorT blackman( WLSampleNrT samples ); /**< Blackman window function. */

    VectorT hanning( WLSampleNrT samples ); /**< Hanning window function. */
} /* namespace WLWindowFunction */

#endif  // WLWINDOWFUNCTION_H_
