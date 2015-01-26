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

#ifndef WLGEOMETRY_H_
#define WLGEOMETRY_H_

#include <cmath>
#include <limits>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>
#include <osg/Array>
#include <osg/ref_ptr>

#include <core/common/math/linearAlgebra/WMatrixFixed.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/data/WLDataTypes.h"
#include "core/data/WLPositions.h"
#include "core/data/enum/WLEExponent.h"

/**
 * Helper functions for geometric computations.
 *
 * \author pieloth
 * \ingroup util
 */
namespace WLGeometry
{
    typedef Eigen::Vector3d Vector3T;
    typedef Eigen::Matrix3Xd PointsT;
    typedef WVector3d Vector;
    typedef WPosition Point;
    typedef double Angle;
    typedef WMatrix3d MatrixRotation;

    bool computeTriangulation( std::vector< WVector3i >* const triangles, osg::ref_ptr< osg::Vec3Array > osgPoints,
                        double transformationFactor = -0.005 );

    bool computeTriangulation( std::vector< WVector3i >* const triangles, const std::vector< WPosition >& positions,
                    double transformationFactor = -0.005 );

    bool computeTriangulation( std::vector< WVector3i >* const triangles, const PointsT& positions,
                        double transformationFactor = -0.005 );

    MatrixRotation getRotationXYZMatrix( double x, double y, double z );

    Point rotate( const MatrixRotation& rotation, const Point& point );

    Vector tranlate( const Vector& translation, const Point& point );

    Point centerOfMass( const std::vector< Point >& );

    double distance( const Point&, const Point& );

    Point minDistance( const Point&, const std::vector< Point >& );

    void transformPoints( std::vector< Point >* const out, const std::vector< Point >& in, const WLMatrix4::Matrix4T& trans );

    bool transformPoints( WLPositions* const out, const WLPositions& in, const WLMatrix4::Matrix4T& trans );

    void toBaseExponent( std::vector< Point >* const out, const std::vector< Point >& in, WLEExponent::Enum exp );

    /**
     * Calculates an orthogonal vector to v.
     *
     * \param o An orthogonal vector of v.
     * \param v Input vector.
     * \return True if o contains an orthogonal vector of v.
     */
    bool findOrthogonalVector( Vector3T* const o, const Vector3T& v );

    /**
     * Calculates an tagent plane for the normal vector n. u, v and n are orthogonal to each other.
     *
     * \param u Plane vector 1 for parametrically description.
     * \param v Plane vector 2 for parametrically description.
     * \param n Normal vector.
     * \return True if u and v contain a plane vector.
     */
    bool findTagentPlane( Vector3T* const u, Vector3T* const v, const Vector3T& n );

    inline bool isAlmostZero( double v, double eps = 1e3 * std::numeric_limits< double >::min() )
    {
        return std::fabs( v ) <= eps;
    }

    /**
     * Creates a half sphere with z >= 0.
     *
     * \param pos Points forming a half sphere.
     * \param points Min. number of points.
     * \param r Radius of the sphere.
     * \return Number of points, which are forming the sphere (=> request points).
     */
    size_t createUpperHalfSphere( PointsT* const pos, size_t points, float r );
}

#endif  // WLGEOMETRY_H_
