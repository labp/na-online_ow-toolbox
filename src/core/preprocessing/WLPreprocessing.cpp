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

#include <string>

#include <Eigen/Dense>   // colPivHouseholderQr

#include <core/common/WLogger.h>

#include "core/data/WLDataTypes.h"

#include "WLPreprocessing.h"

namespace WLPreprocessing
{
    static const std::string NAMESPACE = "WLPreprocessing";

    typedef WLMatrix::MatrixT MatrixT;

    bool baselineCorrection( WLEMData::DataT* const dataOut, const WLEMData::DataT& dataIn, WLSampleIdxT start,
                    WLSampleNrT offset )
    {
        const WLChanNrT channels = dataIn.rows();
        const WLSampleNrT samples = dataIn.cols();

        if( ( start + offset ) > samples )
        {
            wlog::error( NAMESPACE ) << __func__ << ": Index out of bound!";
            *dataOut = dataIn;
            return false;
        }
        if( offset == 0 )
        {
            wlog::warn( NAMESPACE ) << __func__ << ": Offset is zero, no baseline was corrected!";
            *dataOut = dataIn;
            return true;
        }

        WLEMData::SampleT means = dataIn.block( 0, start, channels, offset ).rowwise().mean();
        *dataOut = dataIn.colwise() - means;

        return true;
    }

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

    void windowing( VectorT* const yptr, const VectorT& x, WLWindowFunction::WLEWindow type )
    {
        const VectorT win = WLWindowFunction::window( x.size(), type );
        *yptr = win.cwiseProduct( x );
    }
} /* namespace WLPreProcessing */
