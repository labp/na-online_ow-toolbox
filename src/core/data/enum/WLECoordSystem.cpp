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
#include <core/common/WLogger.h>

#include "WLECoordSystem.h"

namespace WLECoordSystem
{
    ContainerT values()
    {
        ContainerT con;

        con.insert( HEAD );
        con.insert( DEVICE );
        con.insert( AC_PC );

        return con;
    }

    std::string name( Enum val )
    {
        switch( val )
        {
            case HEAD:
                return "Head";
            case DEVICE:
                return "Device";
            case AC_PC:
                return "AC PC";
            case UNKNOWN:
                return "UNKNOWN";
            default:
                WAssert( false, "Unknown WLECoordSystem!" );
                return WLECoordSystem::name( UNKNOWN );
        }
    }

    Enum convertFIFF( WLFiffLib::coord_system_t coord )
    {
        switch( coord )
        {
            case WLFiffLib::CoordSystem::UNKNOWN:
                return UNKNOWN;
            case WLFiffLib::CoordSystem::HEAD:
                return HEAD;
            case WLFiffLib::CoordSystem::DEVICE:
                return DEVICE;
            default:
                wlog::warn( "WLECoordSystem" ) << "No conversion from WLFiffLib::CoordSystem(" << coord << ") to WLECoordSystem!";
                return WLECoordSystem::UNKNOWN;
        }
    }
} /* namespace WLECoordSystem */
