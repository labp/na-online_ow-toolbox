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

#include "WLECardinalPoint.h"

namespace WLECardinalPoint
{

    ContainerT values()
    {
        ContainerT con;

        con.insert( LPA );
        con.insert( NASION );
        con.insert( RPA );

        return con;
    }

    std::string name( Enum val )
    {
        switch( val )
        {
            case LPA:
                return "Left pre-auricular";
            case NASION:
                return "Nasion";
            case RPA:
                return "Right pre-auricular";
            default:
                WAssert( false, "Unknown WLECardinalPoint!" );
                return "ERROR: Undefined!";
        }
    }
}
