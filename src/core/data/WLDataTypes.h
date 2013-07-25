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

#include <boost/shared_ptr.hpp>

// NOTE: Needs Eigen v3.1 or higher for sparse matrices, see README
#include <Eigen/Core>
#include <Eigen/SparseCore>

#ifndef WLDATATYPES_H_
#define WLDATATYPES_H_

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::SparseMatrix;

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
    typedef Matrix< ScalarT, Dynamic, 1 > VectorT;
    typedef boost::shared_ptr< VectorT > SPtr;
    typedef boost::shared_ptr< const VectorT > ConstSPtr;
}

namespace WLRowVector
{
    typedef Matrix< ScalarT, 1, Dynamic > RowVectorT;
    typedef boost::shared_ptr< RowVectorT > SPtr;
    typedef boost::shared_ptr< const RowVectorT > ConstSPtr;
}

namespace WLMatrix
{
    typedef Matrix< ScalarT, Dynamic, Dynamic > MatrixT;
    typedef boost::shared_ptr< MatrixT > SPtr;
    typedef boost::shared_ptr< const MatrixT > ConstSPtr;
}

/**
 * Replacement for sparse matrix type. E.g. SparseMatrix< double >
 */
namespace WLSpMatrix
{
    typedef SparseMatrix< ScalarT > SpMatrixT;
    typedef boost::shared_ptr< SpMatrixT > SPtr;
    typedef boost::shared_ptr< const SpMatrixT > ConstSPtr;
}

#endif  // WLDATATYPES_H_
