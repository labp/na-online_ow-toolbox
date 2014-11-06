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
 * TODO(pieloth): documentation
 *
 * \author pieloth
 */
namespace WLWindowFunction
{
    typedef WLVector::VectorT VectorT;

    enum WLEWindow
    {
        HAMMING, RECTANGLE, BARLETT, BLACKMAN, HANNING, UNKNOWN
    };

    std::set< WLEWindow > values();

    std::string name( WLEWindow value );

    VectorT window( WLSampleNrT samples, WLEWindow type );

    VectorT hamming( WLSampleNrT samples );

    VectorT rectangle( WLSampleNrT samples );

    VectorT barlett( WLSampleNrT samples );

    VectorT blackman( WLSampleNrT samples );

    VectorT hanning( WLSampleNrT samples );
} /* namespace WLWindowFunction */

#endif  // WLWINDOWFUNCTION_H_
