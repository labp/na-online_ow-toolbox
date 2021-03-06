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

#include <cmath>  // abs()
#include <string>

#include <Eigen/Dense>  // min/max

#include <fiff/fiff_dir_entry.h>
#include <fiff/fiff_dir_tree.h>
#include <fiff/fiff_stream.h>
#include <fiff/fiff_types.h>
#include <mne/mne_sourcespace.h>
#include <mne/mne_hemisphere.h>

#include <QtCore/QFile>
#include <QtCore/QList>
#include <QtCore/QString>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>

#include "core/container/WLArrayList.h"

#include "WLReaderSourceSpace.h"

using std::string;
using std::vector;
using namespace FIFFLIB;
using namespace MNELIB;

const std::string WLReaderSourceSpace::CLASS = "WLReaderSourceSpace";

WLReaderSourceSpace::WLReaderSourceSpace( std::string fname ) throw( WDHNoSuchFile ) :
                WLReaderGeneric< WLEMMSurface::SPtr >( fname )
{
}

WLReaderSourceSpace::~WLReaderSourceSpace()
{
}

WLIOStatus::IOStatusT WLReaderSourceSpace::read( WLEMMSurface::SPtr* const surface )
{
    // Reading MNE type
    QFile file( m_fname.c_str() );
    FiffStream::SPtr fiffStream( new FiffStream( &file ) );
    FiffDirTree fiffDirTree;
    QList< FiffDirEntry > fiffDirEntries;

    if( !fiffStream->open( fiffDirTree, fiffDirEntries ) )
    {
        wlog::error( CLASS ) << "Could not open FIFF stream!";
        return WLIOStatus::ERROR_FOPEN;
    }

    QList< FiffDirTree > srcSpaces = fiffDirTree.dir_tree_find( FIFFB_MNE_SOURCE_SPACE );
    if( srcSpaces.empty() )
    {
        wlog::error( CLASS ) << "No source spaces available!";
        fiffStream->device()->close();
        return WLIOStatus::ERROR_UNKNOWN;
    }
    wlog::debug( CLASS ) << "srcSpaces: " << srcSpaces.size();

    MNESourceSpace sourceSpace; // = NULL;
    if( !MNESourceSpace::readFromStream( fiffStream, true, fiffDirTree, sourceSpace ) )
    {
        fiffStream->device()->close();
        wlog::error( CLASS ) << "Could not read the source spaces";
        return WLIOStatus::ERROR_FREAD;
    }

    for( qint32 k = 0; k < sourceSpace.size(); ++k )
    {
        sourceSpace[k].id = MNESourceSpace::find_source_space_hemi( sourceSpace[k] );
        wlog::debug( CLASS ) << "ID: " << sourceSpace[k].id;
        wlog::debug( CLASS ) << "Sources: " << sourceSpace[k].np;
        wlog::debug( CLASS ) << "Faces: " << sourceSpace[k].ntri;
        wlog::debug( CLASS ) << "Coord Frame: " << sourceSpace[k].coord_frame;
    }
    if( sourceSpace.size() != 2 )
    {
        wlog::error( CLASS ) << "Missing one or all hemispheres!";
        return WLIOStatus::ERROR_UNKNOWN;
    }

    if( !( *surface ) )
    {
        wlog::debug( CLASS ) << "No surface instance! Creating a new one.";
        surface->reset( new WLEMMSurface() );
    }
    WLEMMSurface* const pSurface = surface->get();

    // Convert to LaBP type
    pSurface->setHemisphere( WLEMMSurface::Hemisphere::BOTH );
    const QString LH = "lh";
    const QString RH = "rh";
    WLPositions::SPtr vSurface = pSurface->getVertex();
    vSurface->coordSystem( WLECoordSystem::AC_PC );

    // Append left and right hemispheres: BOTH = LH|RH
    vSurface->resize( sourceSpace[LH].np + sourceSpace[RH].np );
    WLPositions::IndexT vIdx = 0;
    for( fiff_int_t i = 0; i < sourceSpace[LH].np; ++i )
    {
        const WLPositions::PositionT dip( sourceSpace[LH].rr.row( i ).cast< WLPositions::ScalarT >() );
        vSurface->data().col( vIdx ) = dip;
        ++vIdx;
    }
    for( fiff_int_t i = 0; i < sourceSpace[RH].np; ++i )
    {
        const WLPositions::PositionT dip( sourceSpace[RH].rr.row( i ).cast< WLPositions::ScalarT >() );
        vSurface->data().col( vIdx ) = dip;
        ++vIdx;
    }
    WAssert( vIdx == vSurface->size(), "vIdx == vSurface->size()" );
    estimateExponent( vSurface.get() );
    wlog::info( CLASS ) << "Vertices: " << vSurface->size();

    WLArrayList< WVector3i >::SPtr faces( new WLArrayList< WVector3i >() );
    faces->reserve( sourceSpace[LH].ntri + sourceSpace[RH].ntri );
    for( fiff_int_t i = 0; i < sourceSpace[LH].ntri; ++i )
    {
        const int x = sourceSpace[LH].tris( i, 0 );
        const int y = sourceSpace[LH].tris( i, 1 );
        const int z = sourceSpace[LH].tris( i, 2 );
        faces->push_back( WVector3i( x, y, z ) );
    }
    const int triOffset = sourceSpace[LH].np;
    for( fiff_int_t i = 0; i < sourceSpace[RH].ntri; ++i )
    {
        const int x = sourceSpace[RH].tris( i, 0 ) + triOffset;
        const int y = sourceSpace[RH].tris( i, 1 ) + triOffset;
        const int z = sourceSpace[RH].tris( i, 2 ) + triOffset;
        faces->push_back( WVector3i( x, y, z ) );
    }

    pSurface->setFaces( faces );
    wlog::info( CLASS ) << "Faces: " << faces->size();
    return WLIOStatus::SUCCESS;
}

void WLReaderSourceSpace::estimateExponent( WLPositions* const pos )
{
    const WLPositions::ScalarT min = pos->data().row( 0 ).minCoeff();
    const WLPositions::ScalarT max = pos->data().row( 0 ).maxCoeff();
    const WLPositions::ScalarT diff = fabs( max - min );
    pos->unit( WLEUnit::METER );
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
