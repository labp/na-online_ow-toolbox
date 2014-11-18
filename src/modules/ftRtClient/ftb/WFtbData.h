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

#ifndef WFTBDATA_H_
#define WFTBDATA_H_

#include <stdint.h>
#include <string>

#include <message.h>

namespace wftb
{
    typedef uint32_t data_type_t;

    typedef datadef_t DataDefT;
    typedef data_t DataT;

    namespace DataType
    {
        const data_type_t CHAR = DATATYPE_CHAR;
        const data_type_t UINT8 = DATATYPE_UINT8;
        const data_type_t UINT16 = DATATYPE_UINT16;
        const data_type_t UINT32 = DATATYPE_UINT32;
        const data_type_t UINT64 = DATATYPE_UINT64;
        const data_type_t INT8 = DATATYPE_INT8;
        const data_type_t INT16 = DATATYPE_INT16;
        const data_type_t INT32 = DATATYPE_INT32;
        const data_type_t INT64 = DATATYPE_INT64;
        const data_type_t FLOAT32 = DATATYPE_FLOAT32;
        const data_type_t FLOAT64 = DATATYPE_FLOAT64;
        const data_type_t UNKNOWN = DATATYPE_UNKNOWN;

        std::string name( data_type_t type );
    } /* namespace DataType */
} /* namespace wftb */
#endif  // WFTBDATA_H_

