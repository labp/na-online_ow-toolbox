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

#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry> // homogeneous

#include <osg/Array>
#include <osgUtil/DelaunayTriangulator>

#include <core/common/WAssert.h>
#include <core/common/WException.h>
#include <core/common/WLogger.h>
#include <core/common/math/WLinearAlgebraFunctions.h>
#include <core/common/math/linearAlgebra/WMatrixFixed.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include <core/graphicsEngine/WGEUtils.h>

#include "core/util/profiler/WLTimeProfiler.h"

#include "WLGeometry.h"

bool WLGeometry::computeTriangulation( std::vector< WVector3i >* const triangles, osg::ref_ptr< osg::Vec3Array > osgPoints,
                double transformationFactor )
{
    WLTimeProfiler tp( "WLGeometry", __func__ );
    const std::string SOURCE = "WLGeometry::computeTriangulation";

    if( triangles == NULL )
    {
        wlog::error( SOURCE ) << "Output pointer is NULL!";
        return false;
    }
    // Using algorithm of core/graphicsEngine/WGEGeometryUtils.cpp
    if( osgPoints->size() < 3 )
    {
        wlog::error( SOURCE ) << "The Delaunay triangulation needs at least 3 vertices!";
        return false;
    }

    if( transformationFactor != 0.0 )
    {
        // Transform the points as described in the Doxygen description of
        // this function.
        osg::Vec3 centroid;
        for( osg::Vec3Array::const_iterator it = osgPoints->begin(); it != osgPoints->end(); ++it )
        {
            centroid += ( *it );
        }
        centroid /= osgPoints->size();

        for( osg::Vec3Array::iterator it = osgPoints->begin(); it != osgPoints->end(); ++it )
        {
            const double factor = ( ( *it ).z() - centroid.z() ) * transformationFactor + 1.0;
            ( *it ).x() = ( ( *it ).x() - centroid.x() ) * factor + centroid.x();
            ( *it ).y() = ( ( *it ).y() - centroid.y() ) * factor + centroid.y();
        }
    }

    // The osg triangulator sorts the points and returns the triangles with
    // the indizes of the sorted points. Since we don't want to change the
    // sequence of the points, we have to save the original index of each
    // point.
    std::map< osg::Vec3, size_t > map;
    for( size_t index = 0; index < osgPoints->size(); ++index )
    {
        map[( *osgPoints )[index]] = index;
    }

    osg::ref_ptr< osgUtil::DelaunayTriangulator > triangulator;

    try
    {
        triangulator = new osgUtil::DelaunayTriangulator( osgPoints );
        if( !triangulator->triangulate() )
        {
            wlog::error( SOURCE ) << "Something went wrong in triangulation.";
            return false;
        }
    }
    catch( const WException& e )
    {
        wlog::error( SOURCE ) << "Unexpected error: " << e.what() << "\n" << e.getBacktrace();
        return false;
    }
    catch( const std::exception& e )
    {
        wlog::error( SOURCE ) << "Unexpected error: " << e.what();
        return false;
    }
    catch( ... )
    {
        wlog::error( SOURCE ) << "Unexpected error!";
        return false;
    }

    const osg::DrawElementsUInt* const osgTriangles = triangulator->getTriangles();
    wlog::debug( SOURCE ) << "osgTriangles: " << osgTriangles->size();
    WAssertDebug( osgTriangles->size() % 3 == 0, "triangles/3 != 0!" );
    osg::DrawElementsUInt::size_type nbTriangles = osgTriangles->size() / 3;
    triangles->reserve( nbTriangles );

    // Convert the new index of the osgTriangle to the original index stored in map.
    osg::DrawElementsUInt::size_type vertID;
    for( osg::DrawElementsUInt::size_type triangleID = 0; triangleID < nbTriangles; ++triangleID )
    {
        vertID = triangleID * 3;

        WVector3i triangle;
        triangle.x() = map[( *osgPoints )[( *osgTriangles )[vertID + 0]]];
        triangle.y() = map[( *osgPoints )[( *osgTriangles )[vertID + 1]]];
        triangle.z() = map[( *osgPoints )[( *osgTriangles )[vertID + 2]]];

        triangles->push_back( triangle );
    }
    wlog::debug( SOURCE ) << "triangles: " << triangles->size();
    return true;
}

bool WLGeometry::computeTriangulation( std::vector< WVector3i >* const triangles, const std::vector< WPosition >& points,
                double transformationFactor )
{
    WLTimeProfiler tp( "WLGeometry", __func__ );

    osg::ref_ptr< osg::Vec3Array > osgPoints = wge::osgVec3Array( points );
    return computeTriangulation( triangles, osgPoints, transformationFactor );
}

bool WLGeometry::computeTriangulation( std::vector< WVector3i >* const triangles, const PointsT& positions,
                double transformationFactor )
{
    WLTimeProfiler tp( "WLGeometry", __func__ );

    osg::ref_ptr< osg::Vec3Array > osgPoints = osg::ref_ptr< osg::Vec3Array >( new osg::Vec3Array );
    osgPoints->reserve( positions.cols() );
    for( PointsT::Index i = 0; i != positions.cols(); ++i )
    {
        const osg::Vec3 tmp( positions.col( i ).x(), positions.col( i ).y(), positions.col( i ).z() );
        osgPoints->push_back( tmp );
    }
    return computeTriangulation( triangles, osgPoints, transformationFactor );
}

WLGeometry::MatrixRotation WLGeometry::getRotationXYZMatrix( double x, double y, double z )
{
    MatrixRotation rot;

    rot( 0, 0 ) = cos( y ) * cos( z );
    rot( 0, 1 ) = -cos( y ) * sin( z );
    rot( 0, 2 ) = sin( y );

    rot( 1, 0 ) = cos( x ) * sin( z ) + cos( z ) * sin( x ) * sin( y );
    rot( 1, 1 ) = cos( x ) * cos( z ) - sin( x ) * sin( y ) * sin( z );
    rot( 1, 2 ) = -cos( y ) * sin( x );

    rot( 2, 0 ) = sin( x ) * sin( z ) - cos( x ) * cos( z ) * sin( y );
    rot( 2, 1 ) = cos( z ) * sin( x ) + cos( x ) * sin( y ) * sin( z );
    rot( 2, 2 ) = cos( x ) * cos( y );

    return rot;
}

WLGeometry::Point WLGeometry::rotate( const MatrixRotation& rotation, const Point& point )
{
    Point rot = rotation * point;
    return rot;
}

WLGeometry::Point WLGeometry::tranlate( const Vector& translation, const Point& point )
{
    Vector tra = translation + point;
    return tra;
}

WLGeometry::Point WLGeometry::centerOfMass( const std::vector< Point >& points )
{
    Point c;
    c.x() = 0;
    c.y() = 0;
    c.z() = 0;

    for( std::vector< Point >::const_iterator it = points.begin(); it != points.end(); ++it )
    {
        c += *it;
    }

    c = c * ( 1.0 / ( double )points.size() );

    return c;
}

WLGeometry::Point WLGeometry::minDistance( const Point& p, const std::vector< Point >& points )
{
    double dist = std::numeric_limits< double >::max();
    double distTmp = 0;
    Point pTmp;
    for( std::vector< Point >::const_iterator it = points.begin(); it != points.end(); ++it )
    {
        distTmp = distance( p, *it );
        if( distTmp < dist )
        {
            dist = distTmp;
            pTmp = *it;
        }
    }
    return pTmp;
}

double WLGeometry::distance( const Point& p1, const Point& p2 )
{
    double xd = p2.x() - p1.x();
    double yd = p2.y() - p1.y();
    double zd = p2.z() - p1.z();

    return sqrt( xd * xd + yd * yd + zd * zd );
}

void WLGeometry::transformPoints( std::vector< Point >* const out, const std::vector< Point >& in,
                const WLMatrix4::Matrix4T& trans )
{
    out->reserve( in.size() );
#ifdef LABP_FLOAT_COMPUTATION
    WMatrix< double > owTrans( ( Eigen::MatrixXf )trans );
#else
    WMatrix< double > owTrans( ( Eigen::MatrixXd )trans );
#endif
    std::vector< Point >::const_iterator it;
    for( it = in.begin(); it != in.end(); ++it )
    {
        const WPosition p = transformPosition3DWithMatrix4D( owTrans, *it );
        out->push_back( p );
    }
}

bool WLGeometry::transformPoints( WLPositions* const out, const WLPositions& in, const WLMatrix4::Matrix4T& trans )
{
    const std::string SOURCE = "WLGeometry::transformPoints";
    if( out == NULL )
    {
        wlog::error( SOURCE ) << "Out pointer is null!";
        return false;
    }
    // TODO(pieloth): #393 Check from/to in positions and transformation!
    // TODO(pieloth): transformPoints - unit test!
    out->resize( in.size() );
    out->positions() = ( trans * in.positions().colwise().homogeneous() ).block( 0, 0, 3, in.size() );

    return true;
}

void WLGeometry::toBaseExponent( std::vector< Point >* const out, const std::vector< Point >& in, WLEExponent::Enum exp )
{
    out->reserve( in.size() );
    double factor = WLEExponent::factor( exp );
    std::vector< Point >::const_iterator it;
    for( it = in.begin(); it != in.end(); ++it )
    {
        const Point p = *it * factor;
        out->push_back( p );
    }
}

bool WLGeometry::findOrthogonalVector( Vector3T* const o, const Vector3T& v )
{
// 0 = Vx Ox + Vy Oy + Vz Oz
    if( v.isZero( 1e3 * std::numeric_limits< double >::min() ) )
    {
        return false;
    }

// avoid division by zero, pick max(abs) as divisor
    Vector3T::Index i;
    v.cwiseAbs().maxCoeff( &i );
    if( i == 2 ) // z
    {
        // - Vz Oz = Vx Ox + Vy Oy
        //      Ox = Vy
        //      Oy = Vx
        //      Oz = -2(Vx * Vy) / Vz
        o->x() = v.y();
        o->y() = v.x();

        o->z() = ( -2.0 * v.x() * v.y() ) / v.z();
        return true;
    }
    if( i == 1 ) // y
    {
        // - Vy Oy = Vx Ox + Vz Oz
        //      Ox = Vz
        //      Oz = Vx
        //      Oy = -2(Vx * Vz) / Vy
        o->x() = v.z();
        o->z() = v.x();
        o->y() = ( -2.0 * v.x() * v.z() ) / v.y();
        return true;
    }
    if( i == 0 ) // x
    {
        // - Vx Ox = Vy Oy + Vz Oz
        //      Oy = Vz
        //      Oz = Vy
        //      Ox = -2(Vy * Vz) / Vx
        o->y() = v.z();
        o->z() = v.y();
        o->x() = ( -2.0 * v.y() * v.z() ) / v.x();
        return true;
    }
    return false;
}

bool WLGeometry::findTagentPlane( Vector3T* const u, Vector3T* const v, const Vector3T& n )
{
    if( !findOrthogonalVector( u, n ) )
    {
        return false;
    }
    *v = n.cross( *u );
    return true;
}

size_t WLGeometry::createUpperHalfSphere( PointsT* const pos, size_t points, float r )
{
    if( points == 0 )
    {
        return 0;
    }
    if( isAlmostZero( r ) )
    {
        return 0;
    }

    typedef Eigen::ArrayXd ArrayT;
    const ArrayT::Index n_angles = ceil( sqrt( points ) );
    const ArrayT::Index n = ( n_angles * n_angles );

// Initialize angles
    ArrayT theta( n_angles ); // vertical, 0° ... 90°
    ArrayT phi( n_angles ); // horizontal, azimut, 0 ... 360°
    for( ArrayT::Index i = 0; i < n_angles; ++i )
    {
        theta( i ) = ( ( double )( i + 1 ) / ( double )n_angles ) * M_PI_2;
        phi( i ) = ( ( double )i / ( double )n_angles ) * 2.0 * M_PI;
    }

// Initialize output
    pos->resize( Eigen::NoChange, n );

// Generate points
// x = r * sin(theta) * cos(phi);
// y = r * sin(theta) * sin(phi);
// z = r * cos(theta);
    ArrayT::Index element = 0;
    const ArrayT sin_t = theta.sin();
    const ArrayT cos_t = theta.cos();
    const ArrayT sin_p = phi.sin();
    const ArrayT cos_p = phi.cos();
    for( ArrayT::Index t = 0; t < n_angles; ++t )
    {
        const double r_sin_t = r * sin_t( t );
        const double z = r * cos_t( t );
        for( ArrayT::Index p = 0; p < n_angles; ++p )
        {
            ( *pos )( 0, element ) = r_sin_t * cos_p( p );
            ( *pos )( 1, element ) = r_sin_t * sin_p( p );
            ( *pos )( 2, element ) = z;
            ++element;
        }
    }

    WAssertDebug( n == element && pos->cols() == n, "n == element && pos->cols() == n" );
    return n;
}
