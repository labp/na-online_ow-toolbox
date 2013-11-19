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

#ifndef WLGEOMETRY_H_
#define WLGEOMETRY_H_

#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WMatrixFixed.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMEnumTypes.h"

namespace WLGeometry
{
    typedef WVector3d Vector;
    typedef WPosition Point;
    typedef double Angle;
    typedef WMatrix3d MatrixRotation;

    bool computeTriangulation( std::vector< WVector3i >* const triangles, const std::vector< WPosition >& positions,
                    double transformationFactor = -0.005 );

    MatrixRotation getRotationXYZMatrix( double x, double y, double z );

    Point rotate( const MatrixRotation& rotation, const Point& point );

    Vector tranlate( const Vector& translation, const Point& point );

    Point centerOfMass( const std::vector< Point >& );

    double distance( const Point&, const Point& );

    Point minDistance( const Point&, const std::vector< Point >& );

    void transformPoints( std::vector< Point >* const out, const std::vector< Point >& in, const WLMatrix4::Matrix4T& trans );

    void toBaseExponent( std::vector< Point >* const out, const std::vector< Point >& in, LaBP::WEExponent::Enum exp );
}

#endif /* WLGEOMETRY_H_ */
