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

#include <cmath>  // M_PI, cos

#include <core/common/WAssert.h>

#include "WLWindowsFunction.h"

namespace WLWindowsFunction
{
    std::set< WLEWindows > values()
    {
        std::set< WLEWindows > values;
        values.insert( HAMMING );
        values.insert( RECTANGLE );
        values.insert( BARLETT );
        values.insert( BLACKMAN );
        values.insert( HANNING );
        return values;
    }

    std::string name( WLEWindows value )
    {
        switch( value )
        {
            case HAMMING:
                return "Hamming";
            case RECTANGLE:
                return "Rectangle";
            case BARLETT:
                return "Barlett";
            case BLACKMAN:
                return "Blackman";
            case HANNING:
                return "Hanning";
            default:
                WAssert( false, "Unknown WLEWindows!" );
                return "ERROR: Undefined!";
        }
    }

    VectorT windows( WLSampleNrT samples, WLEWindows type )
    {
        switch( type )
        {
            case HAMMING:
                return hamming( samples );
            case BARLETT:
                return barlett( samples );
            case BLACKMAN:
                return blackman( samples );
            case HANNING:
                return hanning( samples );
            case RECTANGLE:
                return rectangle( samples );
            default:
                WAssert( false, "Unknown WLEWindows!" );
                return rectangle( samples );
        }
    }

    VectorT hamming( WLSampleNrT samples )
    {
        VectorT w( samples );
        const VectorT::Index M = w.size();
        const VectorT::Index Mm1 = M - 1;
        for( VectorT::Index n = 0; n < M; ++n )
        {
            w( n ) = 0.54 - ( 0.46 * cos( ( 2.0 * M_PI * n ) / Mm1 ) );
        }
        return w;
    }

    VectorT rectangle( WLSampleNrT samples )
    {
        return VectorT::Ones( samples );
    }

    VectorT barlett( WLSampleNrT samples )
    {
        VectorT w( samples );
        const VectorT::Index M = w.size();
        const VectorT::Index Mm1 = M - 1;
        for( VectorT::Index n = 0; n < M; ++n )
        {
            w( n ) = 1.0 - ( ( 2.0 * fabs( n - ( ( Mm1 ) / 2.0 ) ) ) / ( Mm1 ) );
        }
        return w;
    }

    VectorT blackman( WLSampleNrT samples )
    {
        VectorT w( samples );
        const VectorT::Index M = w.size();
        const VectorT::Index Mm1 = M - 1;
        for( VectorT::Index n = 0; n < M; ++n )
        {
            w( n ) = 0.42 - ( 0.5 * cos( ( 2.0 * M_PI * n ) / ( Mm1 ) ) ) + 0.08 * cos( ( 4.0 * M_PI * n ) / ( Mm1 ) );
        }
        return w;
    }

    VectorT hanning( WLSampleNrT samples )
    {
        VectorT w( samples );
        const VectorT::Index M = w.size();
        const VectorT::Index Mm1 = M - 1;
        for( VectorT::Index n = 0; n < M; ++n )
        {
            w( n ) = 0.5 * ( 1.0 - cos( ( 2.0 * M_PI * n ) / ( Mm1 ) ) );
        }
        return w;
    }
} /* namespace WLWindowsFunction */
