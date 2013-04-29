/**
 * TODO license and documentation
 */

#include <cmath>
#include <map>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <osg/Array>
#include <osgUtil/DelaunayTriangulator>

#include <core/common/WLogger.h>
#include <core/common/math/linearAlgebra/WMatrixFixed.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include <core/graphicsEngine/WGEUtils.h>

#include "WLGeometry.h"

bool WLGeometry::computeTriangulation( std::vector< WVector3i >& triangles, const std::vector< WPosition >& points,
                double transformationFactor )
{
    // Using algorithm of core/graphicsEngine/WGEGeometryUtils.cpp
    if( points.size() < 3 )
    {
        wlog::error( "WGeometry" ) << "computeTriangulation() The Delaunay triangulation needs at least 3 vertices!";
        return false;
    }

    osg::ref_ptr< osg::Vec3Array > osgPoints = wge::osgVec3Array( points );
    wlog::debug( "WGeometry" ) << "computeTriangulation() osgPoints: " << osgPoints->size();

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

    osg::ref_ptr< osgUtil::DelaunayTriangulator > triangulator( new osgUtil::DelaunayTriangulator( osgPoints ) );

    bool triangulationResult = triangulator->triangulate();
    if( !triangulationResult )
    {
        wlog::error( "WGeometry" ) << "computeTriangulation() Something went wrong in triangulation.";
        return false;
    }

    WAssert( triangulationResult, "Something went wrong in triangulation." );

    osg::ref_ptr< const osg::DrawElementsUInt > osgTriangles( triangulator->getTriangles() );
    wlog::debug( "WGeometry" ) << "computeTriangulation() osgTriangles: " << osgTriangles->size();
    size_t nbTriangles = osgTriangles->size() / 3;
    triangles.reserve( nbTriangles );

    // Convert the new index of the osgTriangle to the original index stored in map.
    size_t vertID;
    for( size_t triangleID = 0; triangleID < nbTriangles; ++triangleID )
    {
        vertID = triangleID * 3;

        WVector3i triangle;
        triangle.x() = map[( *osgPoints )[( *osgTriangles )[vertID + 0]]];
        triangle.y() = map[( *osgPoints )[( *osgTriangles )[vertID + 1]]];
        triangle.z() = map[( *osgPoints )[( *osgTriangles )[vertID + 2]]];

        triangles.push_back( triangle );
    }
    wlog::debug( "WGeometry" ) << "computeTriangulation() triangles: " << triangles.size();
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
