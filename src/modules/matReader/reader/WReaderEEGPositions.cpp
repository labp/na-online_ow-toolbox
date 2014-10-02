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
                WLReaderGeneric< std::vector< WPosition > >( fname )
{
}

WReaderEEGPositions::~WReaderEEGPositions()
{
}

WLIOStatus::IOStatusT WReaderEEGPositions::read( std::vector< WPosition >* const positions )
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

    positions->clear();
    QList< FiffChInfo >::Iterator itChInfo;
    size_t skipped = 0;
    for( itChInfo = fiffChInfos.begin(); itChInfo != fiffChInfos.end(); ++itChInfo )
    {
        if( itChInfo->coil_type != FIFFV_COIL_EEG || itChInfo->kind != FIFFV_EEG_CH )
        {
            ++skipped;
            continue;
        }
        const WPosition pos( itChInfo->eeg_loc.col( 0 ).x(), itChInfo->eeg_loc.col( 0 ).y(), itChInfo->eeg_loc.col( 0 ).z() );
        positions->push_back( pos );
    }
    wlog::debug( CLASS ) << "Channels skipped: " << skipped;

    return WLIOStatus::SUCCESS;
}
