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

#ifndef WLDATATYPES_H_
#define WLDATATYPES_H_

#include <boost/shared_ptr.hpp>

// NOTE: Needs Eigen v3.1 or higher for sparse matrices, see README
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "core/dataFormat/fiff/WLFiffLib.h"

/**
 * Replacement for a scalar type used for computations.
 */
#ifdef LABP_FLOAT_COMPUTATION
typedef float ScalarT;
#else
typedef double ScalarT;
#endif  // LABP_FLOAT_COMPUTATION

// See: http://eigen.tuxfamily.org/dox/QuickRefPage.html
// NOTE: Class specialization for SPtr/ConstSPtr binding is not possible, because: MatrixT.row(i) != RowVectorT, MatrixT.row(i) == Eigen::RowVector

namespace WLVector
{
    typedef Eigen::Matrix< ScalarT, Eigen::Dynamic, 1 > VectorT;
    typedef VectorT::Scalar ScalarT;
    typedef boost::shared_ptr< VectorT > SPtr;
    typedef boost::shared_ptr< const VectorT > ConstSPtr;
}

namespace WLRowVector
{
    typedef Eigen::Matrix< ScalarT, 1, Eigen::Dynamic > RowVectorT;
    typedef RowVectorT::Scalar ScalarT;
    typedef boost::shared_ptr< RowVectorT > SPtr;
    typedef boost::shared_ptr< const RowVectorT > ConstSPtr;
}

namespace WLMatrix
{
    typedef Eigen::Matrix< ScalarT, Eigen::Dynamic, Eigen::Dynamic > MatrixT;
    typedef MatrixT::Scalar ScalarT;
    typedef boost::shared_ptr< MatrixT > SPtr;
    typedef boost::shared_ptr< const MatrixT > ConstSPtr;
}

/**
 * 4x4 matrix e.g. transformation matrix.
 */
namespace WLMatrix4
{
    typedef Eigen::Matrix< ScalarT, 4, 4 > Matrix4T;
    typedef Matrix4T::Scalar ScalarT;
    typedef boost::shared_ptr< Matrix4T > SPtr;
    typedef boost::shared_ptr< const Matrix4T > ConstSPtr;
}

/**
 * Replacement for sparse matrix type. E.g. SparseMatrix< double >
 */
namespace WLSpMatrix
{
    typedef Eigen::SparseMatrix< ScalarT > SpMatrixT;
    typedef SpMatrixT::Scalar ScalarT;
    typedef boost::shared_ptr< SpMatrixT > SPtr;
    typedef boost::shared_ptr< const SpMatrixT > ConstSPtr;
}

typedef WLFiffLib::freq_t WLFreqT; /**< Type for frequencies: Hz, kHz and more.*/
typedef WLFiffLib::ident_t WLIdentT; /**< Type for decimal identification, running numbers and more. */
typedef WLFiffLib::ichan_t WLChanIdxT; /**< Index type for channels. */
typedef WLFiffLib::nchan_t WLChanNrT; /**< Type for number of channels (size, count). */
typedef WLFiffLib::isamples_t WLSampleIdxT; /**< Index type for samples. */
typedef WLFiffLib::nsamples_t WLSampleNrT; /**< Type for number of samples (size, count). */
typedef WLFiffLib::time_t WLTimeT; /**< Type for time values: ms, s, min and more. */

#endif  // WLDATATYPES_H_
