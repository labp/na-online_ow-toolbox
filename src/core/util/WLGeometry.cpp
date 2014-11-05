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
#include <Eigen/Core>

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

bool WLGeometry::computeTriangulation( std::vector< WVector3i >* const triangles, const std::vector< WPosition >& points,
                double transformationFactor )
{
    WLTimeProfiler tp( "WLGeometry", "computeTriangulation" );

    const std::string SOURCE = "WLGeometry::computeTriangulation";

    // Using algorithm of core/graphicsEngine/WGEGeometryUtils.cpp
    if( points.size() < 3 )
    {
        wlog::error( SOURCE ) << "The Delaunay triangulation needs at least 3 vertices!";
        return false;
    }

    osg::ref_ptr< osg::Vec3Array > osgPoints = wge::osgVec3Array( points );
    wlog::debug( SOURCE ) << "osgPoints: " << osgPoints->size();

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
    size_t nbTriangles = osgTriangles->size() / 3;
    triangles->reserve( nbTriangles );

    // Convert the new index of the osgTriangle to the original index stored in map.
    size_t vertID;
    for( size_t triangleID = 0; triangleID < nbTriangles; ++triangleID )
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
