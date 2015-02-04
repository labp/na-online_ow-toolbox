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

#include <fiff/fiff.h>
#include <fiff/fiff_ch_info.h>
#include <fiff/fiff_dir_tree.h>
#include <fiff/fiff_stream.h>
#include <fiff/fiff_tag.h>

#include <QtCore/QFile>
#include <QtCore/QList>
#include <QtCore/QString>

#include <core/common/WLogger.h>

#include "WReaderEEGPositions.h"

using namespace FIFFLIB;

const std::string WReaderEEGPositions::CLASS = "WReaderEEGPositions";

WReaderEEGPositions::WReaderEEGPositions( std::string fname ) throw( WDHNoSuchFile ) :
                WLReaderGeneric< WLPositions >( fname )
{
}

WReaderEEGPositions::~WReaderEEGPositions()
{
}

WLIOStatus::IOStatusT WReaderEEGPositions::read( WLPositions* const positions )
{
    QFile file( QString::fromStdString( m_fname ) );

    FiffStream::SPtr fiffStream;
    FiffDirTree dirTree;
    QList< FiffDirEntry > dirEntries;

    if( !Fiff::open( file, fiffStream, dirTree, dirEntries ) )
    {
        wlog::error( CLASS ) << "Could not open fiff file!";
        return WLIOStatus::ERROR_FOPEN;
    }

    QList< FiffDirEntry > fiffChInfEntries;
    QList< FiffDirEntry >::Iterator itEntries;
    for( itEntries = dirEntries.begin(); itEntries != dirEntries.end(); ++itEntries )
    {
        if( itEntries->kind == FIFF_CH_INFO )
        {
            fiffChInfEntries.append( ( *itEntries ) );
        }
    }

    if( fiffChInfEntries.size() == 0 )
    {
        wlog::error( CLASS ) << "No entries for ChannelInfo found!";
        return WLIOStatus::ERROR_FREAD;
    }
    wlog::debug( CLASS ) << "fiffChInfEntries: " << fiffChInfEntries.size();

    QList< FiffChInfo > fiffChInfos;
    FiffTag::SPtr tag;
    for( itEntries = fiffChInfEntries.begin(); itEntries != fiffChInfEntries.end(); ++itEntries )
    {
        const fiff_int_t kind = itEntries->kind;
        const fiff_int_t pos = itEntries->pos;
        if( kind != FIFF_CH_INFO )
        {
            wlog::debug( CLASS ) << "No channel info!";
            continue;
        }

        if( !FiffTag::read_tag( fiffStream.data(), tag, pos ) )
        {
            wlog::error( CLASS ) << "Could not read tag!";
            continue;
        }

        fiffChInfos.append( tag->toChInfo() );
    }

    QList< FiffChInfo >::Iterator itChInfo;
    size_t skipped = 0;
    WLPositions::IndexT nEeg = 0;
    for( itChInfo = fiffChInfos.begin(); itChInfo != fiffChInfos.end(); ++itChInfo )
    {
        if( itChInfo->coil_type != FIFFV_COIL_EEG || itChInfo->kind != FIFFV_EEG_CH )
        {
            ++skipped;
            continue;
        }
        ++nEeg;
    }
    positions->resize(nEeg);
    nEeg = 0;
    for( itChInfo = fiffChInfos.begin(); itChInfo != fiffChInfos.end(); ++itChInfo )
    {
        if( itChInfo->coil_type != FIFFV_COIL_EEG || itChInfo->kind != FIFFV_EEG_CH )
        {
            continue;
        }
        positions->data().col(nEeg).x() = itChInfo->eeg_loc.col( 0 ).x();
        positions->data().col(nEeg).y() = itChInfo->eeg_loc.col( 0 ).y();
        positions->data().col(nEeg).z() = itChInfo->eeg_loc.col( 0 ).z();
    }
    wlog::debug( CLASS ) << "Channels skipped: " << skipped;

    return WLIOStatus::SUCCESS;
}
