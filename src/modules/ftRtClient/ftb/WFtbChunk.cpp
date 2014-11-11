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

#include "WFtbChunk.h"

std::string wftb::ChunkType::name( chunk_type_t type )
{
    switch( type )
    {
        case UNSPECIFIED:
            return "FT_CHUNK_UNSPECIFIED";
        case CHANNEL_NAMES:
            return "FT_CHUNK_CHANNEL_NAMES";
        case CHANNEL_FLAGS:
            return "FT_CHUNK_CHANNEL_FLAGS";
        case RESOLUTIONS:
            return "FT_CHUNK_RESOLUTIONS";
        case ASCII_KEYVAL:
            return "FT_CHUNK_ASCII_KEYVAL";
        case NIFTI1:
            return "FT_CHUNK_NIFTI1";
        case SIEMENS_AP:
            return "FT_CHUNK_SIEMENS_AP";
        case CTF_RES4:
            return "FT_CHUNK_CTF_RES4";
        case NEUROMAG_HEADER:
            return "FT_CHUNK_NEUROMAG_HEADER";
        case NEUROMAG_ISOTRAK:
            return "FT_CHUNK_NEUROMAG_ISOTRAK";
        case NEUROMAG_HPIRESULT:
            return "FT_CHUNK_NEUROMAG_HPIRESULT";
        default:
            return "Unknown!";
    }
}
