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

#include "core/data/WLDataTypes.h"
#include "core/data/WLDigPoint.h"
#include "core/dataFormat/fiff/WLFiffBlockType.h"
#include "core/dataFormat/fiff/WLFiffHPI.h"
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

    const bool hasHpiMeas = readHpiMeas( hpiInfo, stream.data(), tree );
    const bool hasHpiResult = readHpiResult( hpiInfo, stream.data(), tree );

    if( !hasHpiResult && !hasHpiMeas )
    {
        wlog::error( CLASS ) << "No data found!";
        return WLIOStatus::ERROR_UNKNOWN;
    }

    // Read some data
    if( hasHpiResult ^ hasHpiMeas )
    {
        return WLIOStatus::SUCCESS;
    }

    // Read digpoints and/or freqs. Check if size are equals.
    if( !hpiInfo->getDigPoints().empty() && !hpiInfo->getHpiFrequencies().empty() )
    {
        if( hpiInfo->getDigPoints().size() == hpiInfo->getHpiFrequencies().size() )
        {
            return WLIOStatus::SUCCESS;
        }
        else
        {
            wlog::error( CLASS ) << "Digitization points and frequencies does not match!";
            return WLIOStatus::ERROR_UNKNOWN;
        }
    }
    else
    {
        return WLIOStatus::SUCCESS;
    }
}

bool WLReaderHpiInfo::readHpiMeas( WLEMMHpiInfo* const hpiInfo, FIFFLIB::FiffStream* const stream,
                const FIFFLIB::FiffDirTree& tree )
{
    QList< FiffDirTree > hpiMeas = tree.dir_tree_find( FIFFB_HPI_MEAS );
    if( hpiMeas.size() == 0 )
    {
        wlog::error( CLASS ) << "Could not found HPI_MEAS block!";
        return false;
    }

    const bool hasHpiCoil = readHpiCoil( hpiInfo, stream, hpiMeas[0] );
    if( !hasHpiCoil )
    {
        wlog::warn( CLASS ) << "Could not read HPI_COIL information!";
    }

    FiffTag::SPtr tag;
    fiff_int_t kind = -1;
    fiff_int_t pos = -1;
    WLChanNrT nHpiCoil = -1;

    for( qint32 k = 0; k < hpiMeas[0].nent; ++k )
    {
        kind = hpiMeas[0].dir[k].kind;
        pos = hpiMeas[0].dir[k].pos;

        if( kind == FIFF_HPI_NCOIL )
        {
            FiffTag::read_tag( stream, tag, pos );
            nHpiCoil = *tag->toInt();
            wlog::debug( CLASS ) << "nHpiCoil: " << nHpiCoil;
            continue;
        }
    }

    if( hasHpiCoil && nHpiCoil > 0 )
    {
        return hpiInfo->getHpiFrequencies().size() == nHpiCoil;
    }
    else
    {
        return hasHpiCoil || nHpiCoil != -1;
    }
}

bool WLReaderHpiInfo::readHpiCoil( WLEMMHpiInfo* const hpiInfo, FIFFLIB::FiffStream* const stream,
                const FIFFLIB::FiffDirTree& tree )
{
    QList< FiffDirTree > hpiCoils = tree.dir_tree_find( WLFiffLib::BlockType::HPI_COIL );
    if( hpiCoils.size() == 0 )
    {
        wlog::error( CLASS ) << "Could not found HPI_COIL block!";
        return false;
    }

    size_t ndata = 0;
    FiffTag::SPtr tag;
    fiff_int_t kind = -1;
    fiff_int_t pos = -1;
    for( qint32 iBlock = 0; iBlock < hpiCoils.size(); ++iBlock )
    {
        for( qint32 iTag = 0; iTag < hpiCoils[iBlock].nent; ++iTag )
        {
            kind = hpiCoils[iBlock].dir[iTag].kind;
            pos = hpiCoils[iBlock].dir[iTag].pos;

            if( kind == WLFiffLib::HPI::COIL_FREQ )
            {
                FiffTag::read_tag( stream, tag, pos );
                hpiInfo->addHpiFrequency( *tag->toFloat() );
                ++ndata;
                continue;
            }
        }
    }
    return ndata > 0;
}

bool WLReaderHpiInfo::readHpiResult( WLEMMHpiInfo* const hpiInfo, FIFFLIB::FiffStream* const stream,
                const FIFFLIB::FiffDirTree& tree )
{
    QList< FiffDirTree > hpiResult = tree.dir_tree_find( FIFFB_HPI_RESULT );
    if( hpiResult.size() == 0 )
    {
        wlog::error( CLASS ) << "Could not found FIFFB_HPI_RESULT!";
        return WLIOStatus::ERROR_UNKNOWN;
    }

    FiffTag::SPtr tag;
    fiff_int_t kind = -1;
    fiff_int_t pos = -1;
    WLChanNrT nHpiCoil = -1;

    size_t ndata = 0;
    for( qint32 k = 0; k < hpiResult[0].nent; ++k )
    {
        kind = hpiResult[0].dir[k].kind;
        pos = hpiResult[0].dir[k].pos;

        if( kind == FIFF_COORD_TRANS )
        {
            FiffTag::read_tag( stream, tag, pos );
            const FiffCoordTrans trans = tag->toCoordTrans();
            if( trans.from == FIFFV_COORD_DEVICE && trans.to == FIFFV_COORD_HEAD )
            {
                hpiInfo->setDevToHead( trans.trans.cast< double >() );
                ++ndata;
                wlog::info( CLASS ) << "Found transformation device to head:\n" << hpiInfo->getDevToHead();
            }
            else
            {
                wlog::error( CLASS ) << "Transformation has wrong from/to: " << trans.from << "/" << trans.to;
            }
            continue;
        }

        if( kind == FIFF_DIG_POINT )
        {
            // TODO(pieloth): These dig points are not equals with the points from isotrak!?
            // Even after transformation dev->head or head->dev ...
            FiffTag::read_tag( stream, tag, pos );
            const FiffDigPoint fDigPnt = tag->toDigPoint();
            WLDigPoint::PointT pnt( fDigPnt.r[0], fDigPnt.r[1], fDigPnt.r[2] );
            WLDigPoint digPnt( pnt, fDigPnt.kind, fDigPnt.ident );
            if( hpiInfo->addDigPoint( digPnt ) )
            {
                wlog::info( CLASS ) << "Found digitization point: " << digPnt.getPoint();
                ++ndata;
                continue;
            }
        }
    }
    return ndata > 0;
}
