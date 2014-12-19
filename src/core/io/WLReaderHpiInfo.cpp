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

#include <QtCore/QFile>
#include <QtCore/QList>
#include <QtCore/QString>

#include <fiff/fiff_coord_trans.h>
#include <fiff/fiff_dig_point.h>
#include <fiff/fiff_dir_entry.h>
#include <fiff/fiff_dir_tree.h>
#include <fiff/fiff_stream.h>
#include <fiff/fiff_tag.h>

#include <core/common/WLogger.h>

#include "core/data/WLDigPoint.h"
#include "WLReaderHpiInfo.h"

using namespace FIFFLIB;

const std::string WLReaderHpiInfo::CLASS = "WLReaderHpiInfo";

WLReaderHpiInfo::WLReaderHpiInfo( std::string fname ) throw( WDHNoSuchFile ) :
                WLReaderGeneric< WLEMMHpiInfo >( fname )
{
}

WLReaderHpiInfo::~WLReaderHpiInfo()
{
}

WLIOStatus::IOStatusT WLReaderHpiInfo::read( WLEMMHpiInfo* const hpiInfo )
{
    if( hpiInfo == NULL )
    {
        return WLIOStatus::ERROR_UNKNOWN;
    }

    FiffStream::SPtr stream( new FiffStream( new QFile( QString::fromStdString( m_fname ) ) ) );
    FiffDirTree tree;
    QList< FiffDirEntry > tags;

    if( stream->open( tree, tags ) )
    {
        wlog::debug( CLASS ) << "Stream opened.";
    }
    else
    {
        wlog::debug( CLASS ) << "Stream not opened.";
        return WLIOStatus::ERROR_FOPEN;
    }

    QList< FiffDirTree > hpiResult = tree.dir_tree_find( FIFFB_HPI_RESULT );
    if( hpiResult.size() == 0 )
    {
        wlog::error( CLASS ) << "Could not find FIFFB_HPI_RESULT.";
        return WLIOStatus::ERROR_UNKNOWN;
    }

    FiffTag::SPtr t_pTag;
    fiff_int_t kind = -1;
    fiff_int_t pos = -1;

    size_t ndata = 0;
    for( qint32 k = 0; k < hpiResult[0].nent; ++k )
    {
        kind = hpiResult[0].dir[k].kind;
        pos = hpiResult[0].dir[k].pos;

        if( kind == FIFF_COORD_TRANS )
        {
            FiffTag::read_tag( stream.data(), t_pTag, pos );
            const FiffCoordTrans trans = t_pTag->toCoordTrans();
            if( trans.from == FIFFV_COORD_DEVICE && trans.to == FIFFV_COORD_HEAD )
            {
                hpiInfo->setDevToHead( trans.trans.cast< double >() );
                ++ndata;
                wlog::info( CLASS ) << "Found transformation device to head.";
            }
            else
            {
                wlog::error( CLASS ) << "Transformation has wrong from/to: " << trans.from << "/" << trans.to;
            }
            continue;
        }

        if( kind == FIFF_DIG_POINT )
        {
            FiffTag::read_tag( stream.data(), t_pTag, pos );
            const FiffDigPoint fDigPnt = t_pTag->toDigPoint();
            WLDigPoint::PointT pnt( fDigPnt.r[0], fDigPnt.r[1], fDigPnt.r[2] );
            WLDigPoint digPnt( pnt, fDigPnt.kind, fDigPnt.ident );
            if( hpiInfo->addDigPoint( digPnt ) )
            {
                wlog::info( CLASS ) << "Found digitization point.";
                ++ndata;
                continue;
            }
        }
    }

    if( ndata > 0 )
    {
        return WLIOStatus::SUCCESS;
    }
    else
    {
        wlog::error( CLASS ) << "No data found!";
        return WLIOStatus::ERROR_UNKNOWN;
    }
}
