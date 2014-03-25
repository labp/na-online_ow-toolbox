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

#include <core/common/WAssert.h>

#include "WLEFTChunkType.h"

WLEFTChunkType::ContainerT WLEFTChunkType::values()
{
    ContainerT modalities;
    modalities.insert( WLEFTChunkType::FT_CHUNK_UNSPECIFIED );
    modalities.insert( WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES );
    modalities.insert( WLEFTChunkType::FT_CHUNK_CHANNEL_FLAGS );
    modalities.insert( WLEFTChunkType::FT_CHUNK_RESOLUTIONS );
    modalities.insert( WLEFTChunkType::FT_CHUNK_ASCII_KEYVAL );
    modalities.insert( WLEFTChunkType::FT_CHUNK_NIFTI1 );
    modalities.insert( WLEFTChunkType::FT_CHUNK_SIEMENS_AP );
    modalities.insert( WLEFTChunkType::FT_CHUNK_CTF_RES4 );
    modalities.insert( WLEFTChunkType::FT_CHUNK_NEUROMAG_HEADER );
    modalities.insert( WLEFTChunkType::FT_CHUNK_NEUROMAG_ISOTRAK );
    modalities.insert( WLEFTChunkType::FT_CHUNK_NEUROMAG_HPIRESULT );
    return modalities;
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
