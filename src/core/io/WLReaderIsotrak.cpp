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

#include <string>

#include <QtCore/QBuffer>
#include <QtCore/QByteArray>
#include <QtCore/QFile>
#include <QtCore/QList>
#include <QtCore/QString>

#include <fiff/fiff.h>
#include <fiff/fiff_ch_info.h>
#include <fiff/fiff_dir_entry.h>
#include <fiff/fiff_dir_tree.h>
#include <fiff/fiff_stream.h>
#include <fiff/fiff_tag.h>

#include <core/common/WIOTools.h>
#include <core/common/WLogger.h>
#include <core/dataHandler/exceptions/WDHNoSuchFile.h>

#include "core/data/enum/WLEPointType.h"

#include "WLReaderIsotrak.h"

using namespace FIFFLIB;

const std::string WLReaderIsotrak::CLASS = "WReaderNeuromagIsotrak";

WLReaderIsotrak::WLReaderIsotrak( std::string fname )
{
    if( !fileExists( fname ) )
    {
        throw WDHNoSuchFile( fname );
    }
    m_stream.reset( new FiffStream( new QFile( QString::fromStdString( fname ) ) ) );
}

WLReaderIsotrak::WLReaderIsotrak( const char* data, size_t size )
{
    m_stream.reset( new FiffStream( new QBuffer( new QByteArray( data, size ) ) ) );
}

WLReaderIsotrak::~WLReaderIsotrak()
{
}

WLIOStatus::IOStatusT WLReaderIsotrak::read( std::list< WLDigPoint >* const digPoints )
{
    digPoints->clear();
    FiffDirTree tree;
    QList< FiffDirEntry > tags;

    if( m_stream->open( tree, tags ) )
    {
        wlog::debug( CLASS ) << "Stream opened.";
    }
    else
    {
        wlog::debug( CLASS ) << "Stream not opened.";
        return WLIOStatus::ERROR_FOPEN;
    }

    return readDigPoints( digPoints, tree ) ? WLIOStatus::SUCCESS : WLIOStatus::ERROR_FREAD;
}

bool WLReaderIsotrak::readDigPoints( std::list< WLDigPoint >* const out, const FiffDirTree& p_Node )
{
    //
    //   Find the desired blocks
    //
    QList< FiffDirTree > isotrak = p_Node.dir_tree_find( FIFFB_ISOTRAK );

    if( isotrak.size() == 0 )
    {
        wlog::error( CLASS ) << "Could not find Isotrak data.";
        return false;
    }

    //
    //  Read Isotrak data.
    //
    FiffTag::SPtr t_pTag;
    QList< FiffChInfo > chs;
    fiff_int_t kind = -1;
    fiff_int_t pos = -1;

    for( qint32 k = 0; k < isotrak[0].nent; ++k )
    {
        kind = isotrak[0].dir[k].kind;
        pos = isotrak[0].dir[k].pos;
        if( kind == FIFF_DIG_POINT )
        {
            FiffTag::read_tag( m_stream.get(), t_pTag, pos );
            const FiffDigPoint point = t_pTag->toDigPoint();
            out->push_back( createDigPoint( point ) );
        }
    }

    return out->size() > 0;
}

WLDigPoint WLReaderIsotrak::createDigPoint( const FiffDigPoint& fiffDigPoint )
{
    const WLDigPoint::PointT p( fiffDigPoint.r[0], fiffDigPoint.r[1], fiffDigPoint.r[2] );
    const WLDigPoint dig( p, fiffDigPoint.kind, fiffDigPoint.ident );

    return dig;
}

