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

// NOTE: Needs Eigen v3.1 or higher for sparse matrices

#ifndef WLMATRIXTYPES_H_
#define WLMATRIXTYPES_H_

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>
#include <Eigen/SparseCore>

using Eigen::MatrixXf;
using Eigen::SparseMatrix;

namespace LaBP
{
    /**
     * Replacement for element type in matrix.
     */
    typedef float MatrixElementT;

    /**
     * Replacement for fixed matrix type. E.g. Eigen::Matrix or WMatrix
     */
    typedef MatrixXf MatrixT;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< MatrixT > MatrixSPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const MatrixT > MatrixConstSPtr;

    /**
     * Replacement for sparse matrix type. E.g. SparseMatrix< double >
     */
    typedef SparseMatrix< MatrixElementT > SpMatrixT;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< SpMatrixT > SpMatrixSPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const SpMatrixT > SpMatrixConstSPtr;
} // namespace LaBP
#endif  // WLMATRIXTYPES_H_
