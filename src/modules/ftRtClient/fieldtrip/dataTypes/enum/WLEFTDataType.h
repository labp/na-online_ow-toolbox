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

#include <message.h>
#include <ostream>
#include <set>
#include <string>

namespace WLEFTDataType
{
    /**
     * Enumeration for FieldTrip data types.
     */
    enum Enum
    {
        /**
         * One byte character.
         */
        CHAR = 0,    //!< CHAR

        /**
         * 8 bit unsigned integer.
         */
        UINT8 = 1,   //!< UINT8

        /**
         * 16 bit unsigned integer.
         */
        UINT16 = 2,  //!< UINT16

        /**
         * 32 bit unsigned integer.
         */
        UINT32 = 3,  //!< UINT32

        /**
         * 64 bit unsigned integer.
         */
        UINT64 = 4,  //!< UINT64

        /**
         * 8 bit signed integer.
         */
        INT8 = 5,    //!< INT8

        /**
         * 16 bit signed integer.
         */
        INT16 = 6,   //!< INT16

        /**
         * 32 bit signed integer.
         */
        INT32 = 7,   //!< INT32

        /**
         * 64 bit signed integer.
         */
        INT64 = 8,   //!< INT64

        /**
         * 32 bit float.
         */
        FLOAT32 = 9, //!< FLOAT32

        /**
         * 64 bit float => double.
         */
        FLOAT64 = 10, //!< FLOAT64

        /**
         * Unknown or unsupported data type.
         */
        UNKNOWN = -1 //!< UNKNOWN

    };

    typedef std::set< Enum > ContainerT;

    ContainerT values();
    std::string name( Enum val );

    UINT32_T codeByType( Enum val );

    Enum typeByCode( UINT32_T type );

    unsigned int wordSize( Enum val );

    std::ostream& operator<<( std::ostream &strm, const WLEFTDataType::Enum& obj );

} /* namespace WLEFTDataType */

inline std::ostream& WLEFTDataType::operator<<( std::ostream &strm, const WLEFTDataType::Enum& obj )
{
    strm << WLEFTDataType::name( obj );
    return strm;
}

#endif /* WLEFTDATATYPE_H_ */
