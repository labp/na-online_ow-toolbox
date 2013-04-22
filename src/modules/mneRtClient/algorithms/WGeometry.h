/**
 * TODO license and documentation
 */
// TODO(pieloth): Same code in emmMeasurement! DRYS!
#ifndef WGEOMETRY_H_
#define WGEOMETRY_H_

#include <boost/shared_ptr.hpp>

#include "core/common/math/linearAlgebra/WMatrixFixed.h"
#include "core/common/math/linearAlgebra/WPosition.h"
#include "core/common/math/linearAlgebra/WVectorFixed.h"

namespace WGeometry
{
    typedef WVector3d Vector;
    typedef WPosition Point;
    typedef double Angle;
    typedef WMatrix3d MatrixRotation;

    bool computeTriangulation( std::vector< WVector3i >& triangles, const std::vector< WPosition >& positions,
                    double transformationFactor = -0.005 );

    MatrixRotation getRotationXYZMatrix( double x, double y, double z );

    Point rotate( const MatrixRotation& rotation, const Point& point );

    Vector tranlate( const Vector& translation, const Point& point );

    Point centerOfMass( const std::vector< Point >& );

    double distance( const Point&, const Point& );

    Point minDistance( const Point&, const std::vector< Point >& );
}

#endif /* WGEOMETRY_H_ */
