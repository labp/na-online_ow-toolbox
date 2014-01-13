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

#include "WLECoilType.h"

WLECoilType::ContainerT WLECoilType::values()
{
    ContainerT con;

    con.insert( WLECoilType::NONE );
    con.insert( WLECoilType::EEG );
    con.insert( WLECoilType::EEG_BIPOLAR );
    con.insert( WLECoilType::VV_PLANAR_W );
    con.insert( WLECoilType::VV_PLANAR_T1 );
    con.insert( WLECoilType::VV_PLANAR_T2 );
    con.insert( WLECoilType::VV_PLANAR_T3 );
    con.insert( WLECoilType::VV_MAG_W );
    con.insert( WLECoilType::VV_MAG_T1 );
    con.insert( WLECoilType::VV_MAG_T2 );
    con.insert( WLECoilType::VV_MAG_T3 );

    return con;
}

WLECoilType::ContainerT WLECoilType::valuesEEG()
{
    ContainerT con;

    con.insert( WLECoilType::EEG );
    con.insert( WLECoilType::EEG_BIPOLAR );

    return con;
}

WLECoilType::ContainerT WLECoilType::valuesMEG()
{
    ContainerT con;

    con.insert( WLECoilType::VV_PLANAR_W );
    con.insert( WLECoilType::VV_PLANAR_T1 );
    con.insert( WLECoilType::VV_PLANAR_T2 );
    con.insert( WLECoilType::VV_PLANAR_T3 );
    con.insert( WLECoilType::VV_MAG_W );
    con.insert( WLECoilType::VV_MAG_T1 );
    con.insert( WLECoilType::VV_MAG_T2 );
    con.insert( WLECoilType::VV_MAG_T3 );

    return con;
}

WLECoilType::ContainerT WLECoilType::valuesMagnetometer()
{
    ContainerT con;

    con.insert( WLECoilType::VV_MAG_W );
    con.insert( WLECoilType::VV_MAG_T1 );
    con.insert( WLECoilType::VV_MAG_T2 );
    con.insert( WLECoilType::VV_MAG_T3 );

    return con;
}

WLECoilType::ContainerT WLECoilType::valuesGradiometer()
{
    ContainerT con;

    con.insert( WLECoilType::VV_PLANAR_W );
    con.insert( WLECoilType::VV_PLANAR_T1 );
    con.insert( WLECoilType::VV_PLANAR_T2 );
    con.insert( WLECoilType::VV_PLANAR_T3 );

    return con;
}

std::string WLECoilType::name( Enum val )
{
    switch( val )
    {
        case NONE:
            return "(none)";
        case EEG:
            return "EEG";
        case EEG_BIPOLAR:
            return "EEG_BIPOLAR";
        case VV_PLANAR_W:
            return "VV_PLANAR_W";
        case VV_PLANAR_T1:
            return "VV_PLANAR_T1";
        case VV_PLANAR_T2:
            return "VV_PLANAR_T2";
        case VV_PLANAR_T3:
            return "VV_PLANAR_T3";
        case VV_MAG_W:
            return "VV_MAG_W";
        case VV_MAG_T1:
            return "VV_MAG_T1";
        case VV_MAG_T2:
            return "VV_MAG_T2";
        case VV_MAG_T3:
            return "VV_MAG_T3";
        default:
            WAssert( false, "Unknown WLECoilType!" );
            return WLECoilType::name( WLECoilType::NONE );
    }
}

WLECoilType::Enum WLECoilType::fromFIFF( WLFiffLib::coil_type_t val )
{
    switch( val )
    {
        case WLFiffLib::CoilType::NONE:
            return WLECoilType::NONE;
        case WLFiffLib::CoilType::EEG:
            return WLECoilType::EEG;
        case WLFiffLib::CoilType::EEG_BIPOLAR:
            return WLECoilType::EEG_BIPOLAR;
        case WLFiffLib::CoilType::VV_PLANAR_W:
            return WLECoilType::VV_PLANAR_W;
        case WLFiffLib::CoilType::VV_PLANAR_T1:
            return WLECoilType::VV_PLANAR_T1;
        case WLFiffLib::CoilType::VV_PLANAR_T2:
            return WLECoilType::VV_PLANAR_T2;
        case WLFiffLib::CoilType::VV_PLANAR_T3:
            return WLECoilType::VV_PLANAR_T3;
        case WLFiffLib::CoilType::VV_MAG_W:
            return WLECoilType::VV_MAG_W;
        case WLFiffLib::CoilType::VV_MAG_T1:
            return WLECoilType::VV_MAG_T1;
        case WLFiffLib::CoilType::VV_MAG_T2:
            return WLECoilType::VV_MAG_T2;
        case WLFiffLib::CoilType::VV_MAG_T3:
            return WLECoilType::VV_MAG_T3;
        default:
            wlog::warn( "WLECoilType" ) << "No conversion from WLFiffLib::CoilType(" << val << ") to WLECoilType!";
            return WLECoilType::NONE;
    }
}
