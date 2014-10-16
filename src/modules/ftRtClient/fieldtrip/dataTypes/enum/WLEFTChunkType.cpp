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

#include <core/common/WAssert.h>

#include "WLEFTChunkType.h"

WLEFTChunkType::ContainerT WLEFTChunkType::values()
{
    ContainerT chunks;
    chunks.insert( WLEFTChunkType::FT_CHUNK_UNSPECIFIED );
    chunks.insert( WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES );
    chunks.insert( WLEFTChunkType::FT_CHUNK_CHANNEL_FLAGS );
    chunks.insert( WLEFTChunkType::FT_CHUNK_RESOLUTIONS );
    chunks.insert( WLEFTChunkType::FT_CHUNK_ASCII_KEYVAL );
    chunks.insert( WLEFTChunkType::FT_CHUNK_NIFTI1 );
    chunks.insert( WLEFTChunkType::FT_CHUNK_SIEMENS_AP );
    chunks.insert( WLEFTChunkType::FT_CHUNK_CTF_RES4 );
    chunks.insert( WLEFTChunkType::FT_CHUNK_NEUROMAG_HEADER );
    chunks.insert( WLEFTChunkType::FT_CHUNK_NEUROMAG_ISOTRAK );
    chunks.insert( WLEFTChunkType::FT_CHUNK_NEUROMAG_HPIRESULT );
    return chunks;
}

std::string WLEFTChunkType::name( WLEFTChunkType::Enum val )
{
    switch( val )
    {
        case WLEFTChunkType::FT_CHUNK_UNSPECIFIED:
            return "Unspecified";
        case WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES:
            return "Channel Names";
        case WLEFTChunkType::FT_CHUNK_CHANNEL_FLAGS:
            return "Channel Flags";
        case WLEFTChunkType::FT_CHUNK_RESOLUTIONS:
            return "Resolutions";
        case WLEFTChunkType::FT_CHUNK_ASCII_KEYVAL:
            return "ASCII Key/Value";
        case WLEFTChunkType::FT_CHUNK_NIFTI1:
            return "NIFTI-1";
        case WLEFTChunkType::FT_CHUNK_SIEMENS_AP:
            return "Siemens AP";
        case WLEFTChunkType::FT_CHUNK_CTF_RES4:
            return "CTF .res4";
        case WLEFTChunkType::FT_CHUNK_NEUROMAG_HEADER:
            return "Neuromag Header";
        case WLEFTChunkType::FT_CHUNK_NEUROMAG_ISOTRAK:
            return "Neuromag Isotrak";
        case WLEFTChunkType::FT_CHUNK_NEUROMAG_HPIRESULT:
            return "Neuomag HPI Result";
        default:
            WAssert( false, "Unknown WLEFTChunkType!" );
            return WLEFTChunkType::name( WLEFTChunkType::FT_CHUNK_UNSPECIFIED );
    }
}

WLEFTChunkType::ContainerT WLEFTChunkType::valuesPrintable()
{
    ContainerT chunks;
    chunks.insert( WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES );
    chunks.insert( WLEFTChunkType::FT_CHUNK_CHANNEL_FLAGS );
    chunks.insert( WLEFTChunkType::FT_CHUNK_RESOLUTIONS );
    chunks.insert( WLEFTChunkType::FT_CHUNK_ASCII_KEYVAL );
    chunks.insert( WLEFTChunkType::FT_CHUNK_SIEMENS_AP );
    return chunks;
}
