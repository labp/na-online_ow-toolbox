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

#include <QtCore/QFile>
#include <QtCore/QList>
#include <QtCore/QString>

#include <fiff/fiff.h>
#include <fiff/fiff_dig_point.h>
#include <fiff/fiff_dir_tree.h>
#include <fiff/fiff_stream.h>
#include <fiff/fiff_tag.h>

#include <core/common/WLogger.h>

#include "core/io/WLReader.h"

#include "WLReaderDigPoints.h"

using namespace FIFFLIB;

const std::string WLReaderDigPoints::CLASS = "WLReaderDigPoints";

WLReaderDigPoints::WLReaderDigPoints( std::string fname ) :
                WLReader( fname )
{
}

WLReaderDigPoints::~WLReaderDigPoints()
{
}

WLReaderDigPoints::ReturnCode::Enum WLReaderDigPoints::read( std::vector< WLDigPoint >* const out )
{
    QFile file( QString::fromStdString( m_fname ) );

    FiffStream::SPtr fiffStream;
    FiffDirTree dirTree;
    QList< FiffDirEntry > dirEntries;

    if( !Fiff::open( file, fiffStream, dirTree, dirEntries ) )
    {
        wlog::error( CLASS ) << "Could not open fiff file!";
        return ReturnCode::ERROR_FOPEN;
    }

    QList< FiffDirEntry > fiffDigEntries;
    QList< FiffDirEntry >::Iterator itEntries;
    for( itEntries = dirEntries.begin(); itEntries != dirEntries.end(); ++itEntries )
    {
        if( itEntries->kind == FIFF_DIG_POINT )
        {
            fiffDigEntries.append( ( *itEntries ) );
        }
    }
    wlog::debug( CLASS ) << "digPointTags.size(): " << fiffDigEntries.size();

    QList< FiffDigPoint > fiffDigs;
    FiffTag::SPtr tag;
    for( itEntries = fiffDigEntries.begin(); itEntries != fiffDigEntries.end(); ++itEntries )
    {
        const fiff_int_t kind = itEntries->kind;
        const fiff_int_t pos = itEntries->pos;
        if( kind != FIFF_DIG_POINT )
        {
            wlog::debug( CLASS ) << "No dig point!";
            continue;
        }

        if( !FiffTag::read_tag( fiffStream.data(), tag, pos ) )
        {
            wlog::error( CLASS ) << "Could not read tag!";
            continue;
        }

        fiffDigs.append( tag->toDigPoint() );
    }

    wlog::debug( CLASS ) << "digPoints.size(): " << fiffDigs.size();

    out->clear();
    out->reserve( fiffDigs.size() );
    QList< FiffDigPoint >::Iterator itDigs;
    for( itDigs = fiffDigs.begin(); itDigs != fiffDigs.end(); ++itDigs )
    {
        const WLDigPoint::PointT p( itDigs->r[0], itDigs->r[1], itDigs->r[2] );
        const WLDigPoint dig( p, itDigs->kind, itDigs->ident );
        out->push_back( dig );
    }

    wlog::debug( CLASS ) << "out.size(): " << out->size();

    return ReturnCode::SUCCESS;
}
