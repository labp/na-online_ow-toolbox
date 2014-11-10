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

#include "WFtbData.h"

std::string wftb::DataType::name( data_type_t type )
{
    switch( type )
    {
        case CHAR:
            return "CHAR";
        case UINT8:
            return "UINT8";
        case UINT16:
            return "UINT16";
        case UINT32:
            return "UINT32";
        case UINT64:
            return "UINT64";
        case INT8:
            return "INT8";
        case INT16:
            return "INT16";
        case INT32:
            return "INT32";
        case INT64:
            return "INT64";
        case FLOAT32:
            return "FLOAT32";
        case FLOAT64:
            return "FLOAT64";
        case UNKNOWN:
            return "UNKNOWN";
        default:
            return "Undefined!";
    }
}
