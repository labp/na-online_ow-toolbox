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

#ifndef WLEFTCHUNKTYPE_H_
#define WLEFTCHUNKTYPE_H_

#include <ostream>
#include <set>
#include <string>

namespace WLEFTChunkType
{

    enum Enum
    {
        FT_CHUNK_UNSPECIFIED = 0,
        FT_CHUNK_CHANNEL_NAMES = 1,
        FT_CHUNK_CHANNEL_FLAGS = 2,
        FT_CHUNK_RESOLUTIONS = 3,
        FT_CHUNK_ASCII_KEYVAL = 4,
        FT_CHUNK_NIFTI1 = 5,
        FT_CHUNK_SIEMENS_AP = 6,
        FT_CHUNK_CTF_RES4 = 7,
        FT_CHUNK_NEUROMAG_HEADER = 8,
        FT_CHUNK_NEUROMAG_ISOTRAK = 9,
        FT_CHUNK_NEUROMAG_HPIRESULT = 10
    };

    typedef std::set< Enum > ContainerT;

    ContainerT values();
    std::string name( Enum val );

    std::ostream& operator<<( std::ostream &strm, const WLEFTChunkType::Enum& obj );

} /* namespace WLEFTChunkType */

inline std::ostream& WLEFTChunkType::operator<<( std::ostream &strm, const WLEFTChunkType::Enum& obj )
{
    strm << WLEFTChunkType::name( obj );
    return strm;
}

#endif /* WLEFTCHUNKTYPE_H_ */
