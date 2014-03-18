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
