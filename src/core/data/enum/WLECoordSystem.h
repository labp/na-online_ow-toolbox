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

#ifndef WLECOORDSYSTEM_H_
#define WLECOORDSYSTEM_H_

#include <ostream>
#include <set>
#include <string>

#include "core/dataFormat/fiff/WLFiffCoordSystem.h"

/**
 * Enumeration for coordinate systems, compatible with FIFF enum(coord).
 *
 * \author pieloth
 * \ingroup data
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
     * Gets the name of the enum value.
     *
     * \param val WLECoordSystem::Enum
     * \return A string.
     */
    std::string name( Enum val );

    /**
     * Converts a FIFF coord value to a WLECoordSystem enum.
     *
     * \param unit FIFF unitm value
     * \return WLECoordSystem::Enum or WLECoordSystem::UNKNOWN if unknown.
     */
    Enum fromFIFF( WLFiffLib::coord_system_t coord );

    std::ostream& operator<<( std::ostream &strm, const WLECoordSystem::Enum& obj );
} /* namespace WLECoordSystem */

inline std::ostream& WLECoordSystem::operator<<( std::ostream &strm, const WLECoordSystem::Enum& obj )
{
    strm << WLECoordSystem::name( obj );
    return strm;
}

#endif  // WLECOORDSYSTEM_H_
