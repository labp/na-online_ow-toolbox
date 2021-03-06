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

#ifndef WLDATATYPES_H_
#define WLDATATYPES_H_

#include <boost/shared_ptr.hpp>
#include <boost/units/io.hpp>  // operator<< for logging
#include <boost/units/quantity.hpp>
#include <boost/units/systems/si.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "core/dataFormat/fiff/WLFiffLib.h"

/**
 * \typedef ScalarT
 * Replacement for a scalar type used for computations.
 */
#ifdef LABP_FLOAT_COMPUTATION
typedef float ScalarT;
#else
typedef double ScalarT;
#endif  // LABP_FLOAT_COMPUTATION

// NOTE: Class specialization of Eigen types for SPtr/ConstSPtr binding is not possible, because:
//       MatrixT.row(i) != RowVectorT, MatrixT.row(i) == Eigen::RowVector
//       So we group the typedefs in a namespace.

/**
 * Helper definitions for a generic (float/double) vector.
 *
 * \ingroup data
 */
namespace WLVector
{
    typedef Eigen::Matrix< ScalarT, Eigen::Dynamic, 1 > VectorT;
    typedef VectorT::Scalar ScalarT;
    typedef boost::shared_ptr< VectorT > SPtr;
    typedef boost::shared_ptr< const VectorT > ConstSPtr;
}

/**
 * Helper definitions for a generic (float/double) row-vector.
 *
 * \ingroup data
 */
namespace WLRowVector
{
    typedef Eigen::Matrix< ScalarT, 1, Eigen::Dynamic > RowVectorT;
    typedef RowVectorT::Scalar ScalarT;
    typedef boost::shared_ptr< RowVectorT > SPtr;
    typedef boost::shared_ptr< const RowVectorT > ConstSPtr;
}

/**
 * Helper definitions for a generic (float/double) matrix.
 *
 * \ingroup data
 */
namespace WLMatrix
{
    typedef Eigen::Matrix< ScalarT, Eigen::Dynamic, Eigen::Dynamic > MatrixT;
    typedef MatrixT::Scalar ScalarT;
    typedef boost::shared_ptr< MatrixT > SPtr;
    typedef boost::shared_ptr< const MatrixT > ConstSPtr;
}

/**
 * Helper definitions for a generic (float/double) fixed 4x4 matrix.
 *
 * \ingroup data
 */
namespace WLMatrix4
{
    typedef Eigen::Matrix< ScalarT, 4, 4 > Matrix4T;
    typedef Matrix4T::Scalar ScalarT;
    typedef boost::shared_ptr< Matrix4T > SPtr;
    typedef boost::shared_ptr< const Matrix4T > ConstSPtr;
}

/**
 * Helper definitions for a generic (float/double) sparse matrix.
 *
 * \ingroup data
 */
namespace WLSpMatrix
{
    typedef Eigen::SparseMatrix< ScalarT > SpMatrixT;
    typedef SpMatrixT::Scalar ScalarT;
    typedef boost::shared_ptr< SpMatrixT > SPtr;
    typedef boost::shared_ptr< const SpMatrixT > ConstSPtr;
}

namespace WLUnits
{
    const boost::units::si::frequency::unit_type Hz = boost::units::si::hertz;
    const boost::units::si::time::unit_type s = boost::units::si::second;
}

typedef boost::units::quantity< boost::units::si::frequency > WLFreqT; /**< Type for frequencies in Hz. */
typedef WLFiffLib::ident_t WLIdentT; /**< Type for decimal identification, running numbers and more. */
typedef WLFiffLib::ichan_t WLChanIdxT; /**< Index type for channels. */
typedef WLFiffLib::nchan_t WLChanNrT; /**< Type for number of channels (size, count). */
typedef WLFiffLib::isamples_t WLSampleIdxT; /**< Index type for samples. */
typedef WLFiffLib::nsamples_t WLSampleNrT; /**< Type for number of samples (size, count). */
typedef boost::units::quantity< boost::units::si::time > WLTimeT; /**< Type for time values in seconds. */

#endif  // WLDATATYPES_H_
