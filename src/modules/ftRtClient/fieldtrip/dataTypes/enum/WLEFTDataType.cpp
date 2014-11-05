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

#include <string>

#include <core/common/WAssert.h>

#include "WLEFTDataType.h"

WLEFTDataType::ContainerT WLEFTDataType::values()
{
    ContainerT modalities;
    modalities.insert( WLEFTDataType::CHAR );
    modalities.insert( WLEFTDataType::UINT8 );
    modalities.insert( WLEFTDataType::UINT16 );
    modalities.insert( WLEFTDataType::UINT32 );
    modalities.insert( WLEFTDataType::UINT64 );
    modalities.insert( WLEFTDataType::INT8 );
    modalities.insert( WLEFTDataType::INT16 );
    modalities.insert( WLEFTDataType::INT32 );
    modalities.insert( WLEFTDataType::INT64 );
    modalities.insert( WLEFTDataType::FLOAT32 );
    modalities.insert( WLEFTDataType::FLOAT64 );
    return modalities;
}

std::string WLEFTDataType::name( WLEFTDataType::Enum val )
{
    switch( val )
    {
        case WLEFTDataType::CHAR:
            return "CHAR";
        case WLEFTDataType::UINT8:
            return "UINT8";
        case WLEFTDataType::UINT16:
            return "UINT16";
        case WLEFTDataType::UINT32:
            return "UINT32";
        case WLEFTDataType::UINT64:
            return "UINT64";
        case WLEFTDataType::INT8:
            return "INT8";
        case WLEFTDataType::INT16:
            return "INT16";
        case WLEFTDataType::INT32:
            return "INT32";
        case WLEFTDataType::INT64:
            return "INT64";
        case WLEFTDataType::FLOAT32:
            return "FLOAT32";
        case WLEFTDataType::FLOAT64:
            return "FLOAT64";
        case WLEFTDataType::UNKNOWN:
            return "UNKNOWN";
        default:
            WAssert( false, "Unknown WLEFTDataType!" );
            return WLEFTDataType::name( WLEFTDataType::UNKNOWN );
    }
}

UINT32_T WLEFTDataType::codeByType( Enum val )
{
    switch( val )
    {
        case WLEFTDataType::CHAR:
            return DATATYPE_CHAR;
        case WLEFTDataType::UINT8:
            return DATATYPE_UINT8;
        case WLEFTDataType::UINT16:
            return DATATYPE_UINT16;
        case WLEFTDataType::UINT32:
            return DATATYPE_UINT32;
        case WLEFTDataType::UINT64:
            return DATATYPE_UINT64;
        case WLEFTDataType::INT8:
            return DATATYPE_INT8;
        case WLEFTDataType::INT16:
            return DATATYPE_INT16;
        case WLEFTDataType::INT32:
            return DATATYPE_INT32;
        case WLEFTDataType::INT64:
            return DATATYPE_INT64;
        case WLEFTDataType::FLOAT32:
            return DATATYPE_FLOAT32;
        case WLEFTDataType::FLOAT64:
            return DATATYPE_FLOAT64;
        case WLEFTDataType::UNKNOWN:
            return DATATYPE_UNKNOWN;
        default:
            WAssert( false, "Unknown WLEFTDataType!" );
            return WLEFTDataType::codeByType( WLEFTDataType::UNKNOWN );
    }
}

WLEFTDataType::Enum WLEFTDataType::typeByCode( UINT32_T type )
{
    switch( type )
    {
        case DATATYPE_CHAR:
            return WLEFTDataType::CHAR;
        case DATATYPE_UINT8:
            return WLEFTDataType::UINT8;
        case DATATYPE_UINT16:
            return WLEFTDataType::UINT16;
        case DATATYPE_UINT32:
            return WLEFTDataType::UINT32;
        case DATATYPE_UINT64:
            return WLEFTDataType::UINT64;
        case DATATYPE_INT8:
            return WLEFTDataType::INT8;
        case DATATYPE_INT16:
            return WLEFTDataType::INT16;
        case DATATYPE_INT32:
            return WLEFTDataType::INT32;
        case DATATYPE_INT64:
            return WLEFTDataType::INT64;
        case DATATYPE_FLOAT32:
            return WLEFTDataType::FLOAT32;
        case DATATYPE_FLOAT64:
            return WLEFTDataType::FLOAT64;
        default:
            return WLEFTDataType::UNKNOWN;
    }
}

unsigned int WLEFTDataType::wordSize( Enum val )
{
    switch( val )
    {
        case DATATYPE_CHAR:
            return WORDSIZE_CHAR;
        case DATATYPE_UINT8:
        case DATATYPE_INT8:
            return WORDSIZE_INT8;
        case DATATYPE_UINT16:
        case DATATYPE_INT16:
            return WORDSIZE_INT16;
        case DATATYPE_UINT32:
        case DATATYPE_INT32:
            return WORDSIZE_INT32;
        case DATATYPE_UINT64:
        case DATATYPE_INT64:
            return WORDSIZE_INT64;
        case DATATYPE_FLOAT32:
            return WORDSIZE_FLOAT32;
        case DATATYPE_FLOAT64:
            return WORDSIZE_FLOAT64;
        case DATATYPE_UNKNOWN:
            WAssert( false, "Data type not supported!" );
    }
    return 0;
}

