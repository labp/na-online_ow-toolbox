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

#ifndef WALIGNMENT_H_
#define WALIGNMENT_H_

#include <list>
#include <string>
#include <utility> // std::pair
#include <vector>

#include <Eigen/Core>

#include <core/common/math/linearAlgebra/WPosition.h>

#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMeasurement.h"

/**
 * Registration and Alignment implementation.
 * A transformation estimation, initial transformation, will be computed if correspondences are available.
 * With or without this initial transformation the alignment is done with an ICP algorithm.
 *
 * \author pieloth
 */
class WAlignment
{
public:
    typedef std::pair< WPosition, WPosition > CorrespondenceT;
    typedef std::vector< WPosition > PointsT;
    typedef WLMatrix4::Matrix4T TransformationT;

    static const std::string CLASS;

    /**
     * Constant to indicate that the algorithm has not converged.
     */
    static const double NOT_CONVERGED;

    /**
     * Constructor.
     *
     * \param maxInterations Maximum iterations for ICP.
     */
    WAlignment( int maxInterations = 10 );
    virtual ~WAlignment();

    /**
     * Adds a pair of corresponding points for transform estimation.
     * A minimum of correspondences is suggested.
     *
     * \param cor A pair of corresponding points.
     */
    void addCorrespondence( const CorrespondenceT& cor );

    /**
     * Deletes all correspondences.
     */
    void clearCorrespondences();

    /**
     * Computes the alignment and stores the transformation matrix.
     * If correspondences are available, a transformation estimation will be done.
     * If no correspondences are available and the matrix is not zero or not an identity,
     * the matrix will be used as a initial transformation.
     *
     * \param matrix Holds the final transformation.
     * \param from Source points.
     * \param to Target points.
     * \return Fitness score (=>0) if converged or NOT_CONVERGED.
     */
    double align( TransformationT* const matrix, const PointsT& from, const PointsT& to );

private:
    typedef Eigen::Matrix< float, 4, 4 > PCLMatrixT;

    std::list< CorrespondenceT > m_correspondences;

    int m_maxIterations;

    bool estimateTransformation( PCLMatrixT* const matrix );

    double icpAlign( PCLMatrixT* const trans, const PointsT& from, const PointsT& to );
};

#endif  // WALIGNMENT_H_
