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

#ifndef WLECOORDSYSTEM_H_
#define WLECOORDSYSTEM_H_

#include <set>

#include "core/fileFormat/fiff/WLFiffCoordSystem.h"

/**
 * Enumeration for coordinate systems, compatible with FIFF enum(coord).
 *
 * \author pieloth
 */
namespace WLECoordSystem
{
    enum Enum
    {
        UNKNOWN = WLFiffLib::CoordSystem::UNKNOWN,
        HEAD = WLFiffLib::CoordSystem::HEAD,
        DEVICE = WLFiffLib::CoordSystem::DEVICE,
        AC_PC = 1000
    };

    typedef std::set< Enum > ContainerT;

    /**
     * Gets all enum values.
     *
     * \return Container with all enum values.
     */
    ContainerT values();

    /**
     * Converts a FIFF coord value to a WLECoordSystem enum.
     *
     * \param unit FIFF unitm value
     * \return WLECoordSystem::Enum or WLECoordSystem::UNKNOWN if unknown.
     */
    Enum convertFIFF( WLFiffLib::coord_system_t coord );

} /* namespace WLECoordSystem */
#endif  // WLECOORDSYSTEM_H_
