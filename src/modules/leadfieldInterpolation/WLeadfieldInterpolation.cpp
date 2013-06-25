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

#include <cmath>    // pow, sqrt
#include <iostream>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>

#include "core/util/profiler/WLTimeProfiler.h"

#include "WLeadfieldInterpolation.h"

const std::string WLeadfieldInterpolation::CLASS = "WLeadfieldInterpolation";
const int WLeadfieldInterpolation::NEIGHBORS = 4;

using LaBP::MatrixSPtr;
using LaBP::MatrixT;

WLeadfieldInterpolation::WLeadfieldInterpolation()
{
}

WLeadfieldInterpolation::~WLeadfieldInterpolation()
{
}

void WLeadfieldInterpolation::setSensorPositions( PositionsSPtr sensors )
{
    m_posSensors = sensors;
}

void WLeadfieldInterpolation::setHDLeadfieldPosition( PositionsSPtr posHdLeadfield )
{
    m_posHDLeadfield = posHdLeadfield;
}

void WLeadfieldInterpolation::setHDLeadfield( MatrixSPtr leadfield )
{
    m_hdLeadfield = leadfield;
}

bool WLeadfieldInterpolation::interpolate( MatrixSPtr leadfield )
{
    WLTimeProfiler tp( CLASS, "interpolate" );
    // Some error checking //
    if( !m_posHDLeadfield || !m_posSensors || !m_hdLeadfield )
    {
        wlog::error( CLASS ) << "Leadfield, sensor positions or BEM Boundary not set!";
        return false;
    }
    if( static_cast< MatrixT::Index >( m_posHDLeadfield->size() ) != m_hdLeadfield->rows() )
    {
        wlog::error( CLASS ) << "Leadfield rows and leadfield positions do not match!";
        return false;
    }

    if( m_posSensors->size() >= m_posHDLeadfield->size() )
    {
        wlog::error( CLASS ) << "Positions and leadfield do not match!";
        return false;
    }

    if( leadfield->cols() != m_hdLeadfield->cols() )
    {
        wlog::error( CLASS ) << "Leadfield sources do not match!";
        return false;
    }

    // Search nearest neighbors //
    std::vector< NeighborsT > neighbors;
    neighbors.resize( m_posSensors->size() );
    if( !searchNearestNeighbor( &neighbors, *m_posSensors, *m_posHDLeadfield ) )
    {
        wlog::error( CLASS ) << "Error in computing nearest neighbor!";
        return false;
    }

    // interpolate leadfield with nearest neighbors //
    float dist0, dist1, dist2, dist3;
    float lf0, lf1, lf2, lf3;
    WAssert( NEIGHBORS == 4, "NEIGHBORS == 4" );
    for( size_t psIdx = 0; psIdx < m_posSensors->size(); ++psIdx )
    {
        dist0 = sqrt( ( *neighbors[psIdx].squareDistances )[0] );
        dist1 = sqrt( ( *neighbors[psIdx].squareDistances )[1] );
        dist2 = sqrt( ( *neighbors[psIdx].squareDistances )[2] );
        dist3 = sqrt( ( *neighbors[psIdx].squareDistances )[3] );
        for( MatrixT::Index srcIdx = 0; srcIdx < leadfield->cols(); ++srcIdx )
        {
            lf0 = ( *m_hdLeadfield )( ( *neighbors[psIdx].indexNeighbors )[0], srcIdx );
            lf1 = ( *m_hdLeadfield )( ( *neighbors[psIdx].indexNeighbors )[1], srcIdx );
            lf2 = ( *m_hdLeadfield )( ( *neighbors[psIdx].indexNeighbors )[2], srcIdx );
            lf3 = ( *m_hdLeadfield )( ( *neighbors[psIdx].indexNeighbors )[3], srcIdx );

            ( *leadfield )( psIdx, srcIdx ) = ( lf0 / dist0 ) + ( lf1 / dist1 ) + ( lf2 / dist2 ) + ( lf3 / dist3 );
        }
        delete neighbors[psIdx].indexNeighbors;
        delete neighbors[psIdx].squareDistances;
    }

    return true;
}

bool WLeadfieldInterpolation::searchNearestNeighbor( std::vector< NeighborsT >* neighbors, const PositionsT& searchPoints,
                const PositionsT& inputPoints )
{
    WLTimeProfiler tp( CLASS, "searchNearestNeighbor" );

    pcl::PointCloud< pcl::PointXYZ >::Ptr inCloud( new pcl::PointCloud< pcl::PointXYZ > );
    inCloud->reserve( inputPoints.size() );
    for( size_t i = 0; i < inputPoints.size(); ++i )
    {
        inCloud->push_back( pcl::PointXYZ( inputPoints[i].x(), inputPoints[i].y(), inputPoints[i].z() ) );
    }

    pcl::KdTreeFLANN< pcl::PointXYZ > kdtree;
    kdtree.setInputCloud( inCloud );

    for( size_t spIdx = 0; spIdx < searchPoints.size(); ++spIdx )
    {
        ( *neighbors )[spIdx].indexNeighbors = new std::vector< int >( NEIGHBORS );
        ( *neighbors )[spIdx].squareDistances = new std::vector< float >( NEIGHBORS );

        pcl::PointXYZ searchPoint( pcl::PointXYZ( searchPoints[spIdx].x(), searchPoints[spIdx].y(), searchPoints[spIdx].z() ) );
        //wlog::info( CLASS ) << "Search point: " << searchPoint;
        if( kdtree.nearestKSearch( searchPoint, NEIGHBORS, *( ( *neighbors )[spIdx].indexNeighbors ),
                        *( ( *neighbors )[spIdx].squareDistances ) ) != NEIGHBORS )
        {
            wlog::error( CLASS ) << "Error in kdtree.nearestKSearch()";
            return false;
        }
    }

    return true;
}

MatrixSPtr WLeadfieldInterpolation::generateRandomLeadfield( size_t sensors, size_t sources )
{
    WLTimeProfiler tp( CLASS, "generateTestHDLeadfield" );
    MatrixSPtr hdLeadfield( new MatrixT( sensors, sources ) );
    hdLeadfield->setRandom();
    return hdLeadfield;
}

