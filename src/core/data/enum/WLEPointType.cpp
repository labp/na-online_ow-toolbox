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

#include "WLEPointType.h"

namespace WLEPointType
{
    WLEPointType::ContainerT values()
    {
        ContainerT con;

        con.insert( UNKNOWN );
        con.insert( CARDINAL );
        con.insert( HPI );
        con.insert( EEG_ECG );
        con.insert( EXTRA );
        con.insert( HEAD_SURFACE );

        return con;
    }

    std::string name( Enum val )
    {
        switch( val )
        {
            case UNKNOWN:
                return "Unknown";
            case CARDINAL:
                return "Cardinal ";
            case HPI:
                return "HPI";
            case EEG_ECG:
                return "EEG/ECG";
            case EXTRA:
                return "EXTRA";
            case HEAD_SURFACE:
                return "Head Surface";

            default:
                WAssert( false, "Unknown WLEPointType!" );
                return name( UNKNOWN );
        }
    }

    Enum fromFIFF( WLFiffLib::point_type_t val )
    {
        switch( val )
        {
            case WLFiffLib::PointType::CARDINAL:
                return CARDINAL;
            case WLFiffLib::PointType::HPI:
                return HPI;
            case WLFiffLib::PointType::EEG:
                wlog::debug( "WLEPointType" ) << "Kind could be EEG or ECG. Using EEG!";
                return EEG_ECG;
            case WLFiffLib::PointType::EXTRA:
                return EXTRA;
            case WLFiffLib::PointType::HEAD_SURFACE:
                return HEAD_SURFACE;
            default:
                wlog::warn( "WLEPointType" ) << "No conversion from WLFiffLib::PointType(" << val << ") to WLEPointType!";
                return UNKNOWN;
        }
    }
}
