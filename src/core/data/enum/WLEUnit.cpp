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
#include <core/common/WLogger.h>

#include "WLEUnit.h"

WLEUnit::ContainerT WLEUnit::values()
{
    ContainerT con;

    con.insert( WLEUnit::NONE );
    con.insert( WLEUnit::UNITLESS );
    con.insert( WLEUnit::METER );
    con.insert( WLEUnit::VOLT );
    con.insert( WLEUnit::TESLA );
    con.insert( WLEUnit::TESLA_PER_METER );
    con.insert( WLEUnit::SIEMENS_PER_METER );

    return con;
}

std::string WLEUnit::name( Enum val )
{
    switch( val )
    {
        case NONE:
            return "(none)";
        case UNITLESS:
            return "(-)";
        case METER:
            return "m";
        case VOLT:
            return "V";
        case TESLA:
            return "T";
        case TESLA_PER_METER:
            return "T/m";
        case SIEMENS_PER_METER:
            return "S/m";
        default:
            WAssert( false, "Unknown WLEUnit!" );
            return WLEUnit::name( WLEUnit::NONE );
    }
}

WLEUnit::Enum WLEUnit::fromFIFF( WLFiffLib::unit_t unit )
{
    switch( unit )
    {
        case WLFiffLib::Unit::NONE:
            return WLEUnit::NONE;
        case WLFiffLib::Unit::UNITLESS:
            return WLEUnit::UNITLESS;
        case WLFiffLib::Unit::M:
            return WLEUnit::METER;
        case WLFiffLib::Unit::V:
            return WLEUnit::VOLT;
        case WLFiffLib::Unit::T:
            return WLEUnit::TESLA;
        case WLFiffLib::Unit::TM:
            return WLEUnit::TESLA;
        default:
            wlog::warn( "WLEUnit" ) << "No conversion from WLFiffLib::Unit(" << unit << ") to WLEUnit!";
            return WLEUnit::NONE;
    }
}
