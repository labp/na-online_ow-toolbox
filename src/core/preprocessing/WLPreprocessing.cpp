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

#include <Eigen/Dense>   // colPivHouseholderQr

#include "core/data/WLDataTypes.h"

#include "WLPreprocessing.h"

namespace WLPreprocessing
{
    typedef WLMatrix::MatrixT MatrixT;

    void detrend( VectorT* const yptr, const VectorT& x )
    {
        VectorT& y = *yptr;
        y.resize( x.size() );
        const VectorT::Index N = x.size();
        //  Build regressor with linear pieces + DC
        MatrixT a = MatrixT::Ones( N, 2 );
        for( MatrixT::Index i = 0; i < N; ++i )
        {
            a( i, 0 ) = ( MatrixT::Scalar )( i + 1.0 ) / N;
        }
        y = x - a * ( a.colPivHouseholderQr().solve( x ) );   // Remove best fit
    }

    void windowing( VectorT* const yptr, const VectorT& x, WLWindowsFunction::WLEWindows type )
    {
        const VectorT win = WLWindowsFunction::windows( x.size(), type );
        *yptr = win.cwiseProduct( x );
    }
} /* namespace WLPreProcessing */
