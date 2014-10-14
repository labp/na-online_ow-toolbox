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

#include "WLEBemType.h"

namespace WLEBemType
{
    ContainerT values()
    {
        ContainerT con;

        con.insert( UNKNOWN );
        con.insert( UNDEFINED );
        con.insert( BRAIN );
        con.insert( SKULL );
        con.insert( HEAD );
        con.insert( INNER_SKIN );
        con.insert( OUTER_SKIN );
        con.insert( INNER_SKULL );
        con.insert( OUTER_SKULL );

        return con;
    }

    std::string name( Enum val )
    {
        switch( val )
        {
            case BRAIN:
                return "Brain";
            case SKULL:
                return "Skull";
            case HEAD:
                return "Skin";
            case INNER_SKIN:
                return "inner_skin";
            case OUTER_SKIN:
                return "outer_skin";
            case INNER_SKULL:
                return "inner_skull";
            case OUTER_SKULL:
                return "outer_skull";
            case UNKNOWN:
                return "Not known";
            case UNDEFINED:
                return "Undefined";
            default:
                WAssert( false, "Unknown WLEBemType!" );
                return WLEBemType::name( UNKNOWN );
        }
    }

    Enum fromFIFF( WLFiffLib::bem_surf_type_t bem )
    {
        switch( bem )
        {
            case WLFiffLib::BEMSurfType::UNKNOWN:
                return UNKNOWN;
            case WLFiffLib::BEMSurfType::UNKNOWN2:
                return UNDEFINED;
            case WLFiffLib::BEMSurfType::BRAIN:
                return BRAIN;
            case WLFiffLib::BEMSurfType::SKULL:
                return SKULL;
            case WLFiffLib::BEMSurfType::HEAD:
                return HEAD;
            default:
                wlog::warn( "WLEBemType" ) << "No conversion from WLFiffLib::BEMSurfType(" << bem << ") to WLEBemType!";
                return WLEBemType::UNKNOWN;
        }
    }

    Enum fromBND( std::string bemName )
    {
        const ContainerT bems = values();
        ContainerT::const_iterator it;
        for( it = bems.begin(); it != bems.end(); ++it )
        {
            if( bemName.find( WLEBemType::name( *it ) ) != std::string::npos )
            {
                return *it;
            }
        }
        wlog::warn( "WLEBemType" ) << "No conversion from BND bem_name (" << bemName << ") to WLEBemType!";
        return UNKNOWN;
    }
} /* namespace WLEBemType */
