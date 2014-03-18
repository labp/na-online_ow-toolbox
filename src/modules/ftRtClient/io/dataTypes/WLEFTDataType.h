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

#ifndef WLEFTDATATYPE_H_
#define WLEFTDATATYPE_H_

#include <ostream>
#include <set>
#include <string>

#include <message.h>

/**
 * Enumeration for FieldTrip data types.
 */
namespace WLEFTDataType
{
    enum Enum
    {
        CHAR = 0,
        UINT8 = 1,
        UINT16 = 2,
        UINT32 = 3,
        UINT64 = 4,
        INT8 = 5,
        INT16 = 6,
        INT32 = 7,
        INT64 = 8,
        FLOAT32 = 9,
        FLOAT64 = 10,
        UNKNOWN = -1

    };

    typedef std::set< Enum > ContainerT;

    ContainerT values();
    std::string name( Enum val );

    std::ostream& operator<<( std::ostream &strm, const WLEFTDataType::Enum& obj );

} /* namespace WLEFTDataType */

inline std::ostream& WLEFTDataType::operator<<( std::ostream &strm, const WLEFTDataType::Enum& obj )
{
    strm << WLEFTDataType::name( obj );
    return strm;
}

#endif /* WLEFTDATATYPE_H_ */
