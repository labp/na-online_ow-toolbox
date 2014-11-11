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

#ifndef WFTBCHUNK_H_
#define WFTBCHUNK_H_

#include <stdint.h>
#include <string>

#include <message.h>

namespace wftb
{
    typedef uint32_t chunk_type_t;
    typedef uint32_t chunk_size_t;

    typedef ft_chunkdef_t ChunkDefT;
    typedef ft_chunk_t ChunkT;

    namespace ChunkType
    {
        const chunk_type_t UNSPECIFIED = FT_CHUNK_UNSPECIFIED;
        const chunk_type_t CHANNEL_NAMES = FT_CHUNK_CHANNEL_NAMES;
        const chunk_type_t CHANNEL_FLAGS = FT_CHUNK_CHANNEL_FLAGS;
        const chunk_type_t RESOLUTIONS = FT_CHUNK_RESOLUTIONS;
        const chunk_type_t ASCII_KEYVAL = FT_CHUNK_ASCII_KEYVAL;
        const chunk_type_t NIFTI1 = FT_CHUNK_NIFTI1;
        const chunk_type_t SIEMENS_AP = FT_CHUNK_SIEMENS_AP;
        const chunk_type_t CTF_RES4 = FT_CHUNK_CTF_RES4;
        const chunk_type_t NEUROMAG_HEADER = FT_CHUNK_NEUROMAG_HEADER;
        const chunk_type_t NEUROMAG_ISOTRAK = FT_CHUNK_NEUROMAG_ISOTRAK;
        const chunk_type_t NEUROMAG_HPIRESULT = FT_CHUNK_NEUROMAG_HPIRESULT;

        std::string name( chunk_type_t type );
    } /* namespace ChunkType */
} /* namespace wftb */
#endif  // WFTBCHUNK_H_
