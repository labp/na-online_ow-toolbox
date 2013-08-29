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

class WAlignment
{
public:
    typedef std::pair< WPosition, WPosition > CorrespondenceT;
    typedef std::vector< WPosition > PointsT;
    typedef WLMatrix4::Matrix4T TransformationT;

    static const std::string CLASS;

    static const double NOT_CONVERGED;

    WAlignment(int maxInterations = 10);
    virtual ~WAlignment();

    void addCorrespondence( const CorrespondenceT& cor );
    void clearCorrespondences();

    double align( TransformationT* const matrix, const PointsT& from, const PointsT& to );

private:
    typedef Eigen::Matrix< float, 4, 4 > PCLMatrixT;

    std::list< CorrespondenceT > m_correspondences;

    int m_maxIterations;

    bool estimateTransformation( PCLMatrixT* const matrix );

    double icpAlign( PCLMatrixT* const trans, const PointsT& from, const PointsT& to );
};

#endif  // WALIGNMENT_H_
