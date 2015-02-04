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

#include <cmath>    // pow, sqrt
#include <iostream>
#include <string>
#include <vector>

#include <fiff/fiff_info.h>
#include <fiff/fiff_ch_info.h>
#include <mne/mne_forwardsolution.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <QList>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>

#include "core/util/profiler/WLTimeProfiler.h"

#include "WLeadfieldInterpolation.h"

const std::string WLeadfieldInterpolation::CLASS = "WLeadfieldInterpolation";
const int WLeadfieldInterpolation::NEIGHBORS = 4;

using WLMatrix::MatrixT;

WLeadfieldInterpolation::WLeadfieldInterpolation()
{
}

WLeadfieldInterpolation::~WLeadfieldInterpolation()
{
}

bool WLeadfieldInterpolation::prepareHDLeadfield( MNELIB::MNEForwardSolution::ConstSPtr hdLeadfield )
{
    WLTimeProfiler tp( CLASS, __func__ );

    const QList< FIFFLIB::FiffChInfo >::size_type num_pos = hdLeadfield->info.chs.size();
    if( num_pos == 0 )
    {
        wlog::error( CLASS ) << "Leadfield sensors/channels are empty!";
        return false;
    }
    const FIFFLIB::fiff_int_t coordFrame = hdLeadfield->info.chs.at( 0 ).coord_frame;
    // NOTE: Head should be fine, because we are using virtual EEG sensors from BEM (AC-PC),
    //       but for mne_forward it seems to be ordinary EEG (HEAD). See nao_eeg_sensor_generator.
    if( coordFrame != WLFiffLib::CoordSystem::HEAD && coordFrame != WLFiffLib::CoordSystem::DATA_VOLUME )
    {
        wlog::error( CLASS ) << "Coord. system should be HEAD or AC-PC! coordFrame=" << coordFrame;
        return false;
    }

    PositionsT::SPtr positions = PositionsT::instance();
    positions->resize( num_pos );

    for( QList< FIFFLIB::FiffChInfo >::size_type ch = 0; ch < num_pos; ++ch )
    {
        const FIFFLIB::FiffChInfo& chInfo = hdLeadfield->info.chs.at( ch );
        if( coordFrame != chInfo.coord_frame )
        {
            wlog::error( CLASS ) << "Coord frames are different: " << coordFrame << "!=" << chInfo.coord_frame;
            return false;
        }
        positions->data().col( ch ).x() = chInfo.loc( 0, 0 );
        positions->data().col( ch ).y() = chInfo.loc( 1, 0 );
        positions->data().col( ch ).z() = chInfo.loc( 2, 0 );
    }
    positions->coordSystem( WLECoordSystem::AC_PC );
    estimateExponent( positions.get() );

    setHDLeadfieldPosition( positions );

#ifdef LABP_FLOAT_COMPUTATION
    WLMatrix::SPtr hdMatrix( new MatrixT( hdLeadfield->sol->data.cast< MatrixT::Scalar >() ) );
#else
    WLMatrix::SPtr hdMatrix( new MatrixT( hdLeadfield->sol->data ) );
#endif  // LABP_FLOAT_COMPUTATION
    setHDLeadfield( hdMatrix );

    return true;
}

void WLeadfieldInterpolation::setSensorPositions( PositionsT::ConstSPtr sensors )
{
    m_posSensors = sensors;
}

void WLeadfieldInterpolation::setHDLeadfieldPosition( PositionsT::ConstSPtr posHdLeadfield )
{
    m_posHDLeadfield = posHdLeadfield;
}

void WLeadfieldInterpolation::setHDLeadfield( WLMatrix::SPtr leadfield )
{
    m_hdLeadfield = leadfield;
}

bool WLeadfieldInterpolation::interpolate( WLMatrix::SPtr leadfield )
{
    WLTimeProfiler tp( CLASS, __func__ );
    // Some error checking //
    if( !m_posHDLeadfield || !m_posSensors || !m_hdLeadfield )
    {
        wlog::error( CLASS ) << "Leadfield, sensor positions or leadfield postions not set!";
        return false;
    }
    if( static_cast< MatrixT::Index >( m_posHDLeadfield->size() ) != m_hdLeadfield->rows() )
    {
        wlog::error( CLASS ) << "Leadfield rows and leadfield positions do not match!";
        return false;
    }

    if( m_posSensors->size() > m_posHDLeadfield->size() )
    {
        wlog::error( CLASS ) << "Sensor and leadfield postions do not match: " << m_posSensors->size() << ">"
                        << m_posHDLeadfield->size();
        return false;
    }
    if( !m_posSensors->isCompatible( *m_posHDLeadfield ) )
    {
        wlog::error( CLASS ) << "Sensor and leadfield positions are not compatible!";
        return false;
    }

    if( leadfield->cols() != m_hdLeadfield->cols() )
    {
        wlog::error( CLASS ) << "Leadfield sources do not match: " << leadfield->cols() << "!=" << m_hdLeadfield->cols();
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
    float w0, w1, w2, w3, w;
    float lf0, lf1, lf2, lf3;

    WAssert( NEIGHBORS == 4, "NEIGHBORS == 4" );
    for( PositionsT::IndexT psIdx = 0; psIdx < m_posSensors->size(); ++psIdx )
    {
        w0 = 1 / sqrt( ( *neighbors[psIdx].squareDistances )[0] );
        w1 = 1 / sqrt( ( *neighbors[psIdx].squareDistances )[1] );
        w2 = 1 / sqrt( ( *neighbors[psIdx].squareDistances )[2] );
        w3 = 1 / sqrt( ( *neighbors[psIdx].squareDistances )[3] );
        w = w0 + w1 + w2 + w3;
        for( MatrixT::Index srcIdx = 0; srcIdx < leadfield->cols(); ++srcIdx )
        {
            lf0 = ( *m_hdLeadfield )( ( *neighbors[psIdx].indexNeighbors )[0], srcIdx );
            lf1 = ( *m_hdLeadfield )( ( *neighbors[psIdx].indexNeighbors )[1], srcIdx );
            lf2 = ( *m_hdLeadfield )( ( *neighbors[psIdx].indexNeighbors )[2], srcIdx );
            lf3 = ( *m_hdLeadfield )( ( *neighbors[psIdx].indexNeighbors )[3], srcIdx );

            ( *leadfield )( psIdx, srcIdx ) = lf0 * ( w0 / w ) + lf1 * ( w1 / w ) + lf2 * ( w2 / w ) + lf3 * ( w3 / w );
        }
        delete neighbors[psIdx].indexNeighbors;
        delete neighbors[psIdx].squareDistances;
    }

    return true;
}

bool WLeadfieldInterpolation::searchNearestNeighbor( std::vector< NeighborsT >* const neighbors, const PositionsT& searchPoints,
                const PositionsT& inputPoints )
{
    WLTimeProfiler tp( CLASS, __func__ );

    pcl::PointCloud< pcl::PointXYZ >::Ptr inCloud( new pcl::PointCloud< pcl::PointXYZ > );
    inCloud->reserve( inputPoints.size() );
    for( PositionsT::IndexT i = 0; i < inputPoints.size(); ++i )
    {
        const PositionsT::PositionT tmp = inputPoints.at( i );
        inCloud->push_back( pcl::PointXYZ( tmp.x(), tmp.y(), tmp.z() ) );
    }

    pcl::KdTreeFLANN < pcl::PointXYZ > kdtree;
    kdtree.setInputCloud( inCloud );

    for( PositionsT::IndexT spIdx = 0; spIdx < searchPoints.size(); ++spIdx )
    {
        ( *neighbors )[spIdx].indexNeighbors = new std::vector< int >( NEIGHBORS );
        ( *neighbors )[spIdx].squareDistances = new std::vector< float >( NEIGHBORS );

        const PositionsT::PositionT tmp = searchPoints.at( spIdx );
        const pcl::PointXYZ searchPoint( tmp.x(), tmp.y(), tmp.z() );
        //wlog::info( CLASS ) << "Search point: " << searchPoint;
        const int found = kdtree.nearestKSearch( searchPoint, NEIGHBORS, *( ( *neighbors )[spIdx].indexNeighbors ),
                        *( ( *neighbors )[spIdx].squareDistances ) );
        if( found != NEIGHBORS )
        {
            wlog::error( CLASS ) << "Error in kdtree.nearestKSearch()";
            return false;
        }
    }

    return true;
}

WLMatrix::SPtr WLeadfieldInterpolation::generateRandomLeadfield( size_t sensors, size_t sources )
{
    WLTimeProfiler tp( CLASS, __func__ );
    WLMatrix::SPtr hdLeadfield( new MatrixT( sensors, sources ) );
    hdLeadfield->setRandom();
    return hdLeadfield;
}

void WLeadfieldInterpolation::estimateExponent( WLPositions* const pos )
{
    const WLPositions::ScalarT min = pos->data().row( 0 ).minCoeff();
    const WLPositions::ScalarT max = pos->data().row( 0 ).maxCoeff();
    const WLPositions::ScalarT diff = fabs( max - min );
    pos->unit( WLEUnit::METER );
    // Assuming a "big" head diameter of 0.5 meter
    if( diff < 0.5 )
    {
        pos->exponent( WLEExponent::BASE );
        wlog::debug( CLASS ) << __func__ << ": estimate meter";
        return;
    }
    if( diff < 50 )
    {
        pos->exponent( WLEExponent::CENTI );
        wlog::debug( CLASS ) << __func__ << ": estimate centimeter";
        return;
    }
    if( diff < 500 )
    {
        pos->exponent( WLEExponent::MILLI );
        wlog::debug( CLASS ) << __func__ << ": estimate millimeter";
        return;
    }
    pos->exponent( WLEExponent::UNKNOWN );
    wlog::debug( CLASS ) << __func__ << ": estimate unknown!";
}
