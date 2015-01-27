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

#include "WLEExponent.h"

WLEExponent::ContainerT WLEExponent::values()
{
    ContainerT con;

    con.insert( WLEExponent::KILO );
    con.insert( WLEExponent::BASE );
    con.insert( WLEExponent::CENTI );
    con.insert( WLEExponent::MILLI );
    con.insert( WLEExponent::MICRO );
    con.insert( WLEExponent::NANO );
    con.insert( WLEExponent::PICO );
    con.insert( WLEExponent::FEMTO );

    return con;
}

std::string WLEExponent::name( Enum val )
{
    switch( val )
    {
        case WLEExponent::KILO:
            return "10^3";
        case WLEExponent::BASE:
            return "1";
        case WLEExponent::CENTI:
            return "10^-2";
        case WLEExponent::MILLI:
            return "10^-3";
        case WLEExponent::MICRO:
            return "10^-6";
        case WLEExponent::NANO:
            return "10^-9";
        case WLEExponent::PICO:
            return "10^-12";
        case WLEExponent::FEMTO:
            return "10^-15";
        case WLEExponent::UNKNOWN:
            return "UNKNOWN";
        default:
            WAssert( false, "Unknown WLEExponent!" );
            return WLEExponent::name( WLEExponent::UNKNOWN );
    }
}

double WLEExponent::factor( Enum val )
{
    switch( val )
    {
        case WLEExponent::KILO:
            return 1.0e+3;
        case WLEExponent::BASE:
            return 1.0;
        case WLEExponent::CENTI:
            return 1.0e-2;
        case WLEExponent::MILLI:
            return 1.0e-3;
        case WLEExponent::MICRO:
            return 1.0e-6;
        case WLEExponent::NANO:
            return 1.0e-9;
        case WLEExponent::PICO:
            return 1.0e-12;
        case WLEExponent::FEMTO:
            return 1.0e-15;
        case WLEExponent::UNKNOWN:
            wlog::warn( "WLEExponent" ) << "Using 1.0 for " << WLEExponent::UNKNOWN;
            return 1.0;
        default:
            WAssert( false, "Unknown WLEExponent!" );
            return WLEExponent::factor( WLEExponent::UNKNOWN );
    }
}

WLEExponent::Enum WLEExponent::fromFIFF( WLFiffLib::unitm_t unitm )
{
    switch( unitm )
    {
        case WLFiffLib::UnitMultiplier::K:
            return WLEExponent::KILO;
        case WLFiffLib::UnitMultiplier::NONE:
            return WLEExponent::BASE;
        case WLFiffLib::UnitMultiplier::C:
            return WLEExponent::CENTI;
        case WLFiffLib::UnitMultiplier::M:
            return WLEExponent::MILLI;
        case WLFiffLib::UnitMultiplier::MU:
            return WLEExponent::MICRO;
        case WLFiffLib::UnitMultiplier::N:
            return WLEExponent::NANO;
        case WLFiffLib::UnitMultiplier::P:
            return WLEExponent::PICO;
        case WLFiffLib::UnitMultiplier::F:
            return WLEExponent::FEMTO;
        default:
            wlog::warn( "WLEExponent" ) << "No conversion from WLFiffLib::UnitMultiplier(" << unitm << ") to WLEExponent!";
            return WLEExponent::UNKNOWN;
    }
}
