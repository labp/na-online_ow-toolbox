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

#ifndef WLEBEMTYPE_H_
#define WLEBEMTYPE_H_

#include <ostream>
#include <set>
#include <string>

#include "core/fileFormat/fiff/WLFiffBEMSurfType.h"

/**
 * Enumeration for BEM surfaces, compatible with FIFF enum(bem_surf_id).
 *
 * \author pieloth
 */
namespace WLEBemType
{
    enum Enum
    {
        UNKNOWN = WLFiffLib::BEMSurfType::UNKNOWN,
        UNDEFINED = WLFiffLib::BEMSurfType::UNKNOWN2,
        BRAIN = WLFiffLib::BEMSurfType::BRAIN,
        SKULL = WLFiffLib::BEMSurfType::SKULL,
        HEAD = WLFiffLib::BEMSurfType::HEAD,
        INNER_SKIN = 1000,
        OUTER_SKIN = 1001,
        INNER_SKULL = 1002,
        OUTER_SKULL = 1003
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
     * \param val WLEBemType::Enum
     * \return A string.
     */
    std::string name( Enum val );

    /**
     * Converts a FIFF bem_surf_id value to a WLEBemType enum.
     *
     * \param unit FIFF unitm value
     * \return WLEBemType::Enum or WLEBemType::UNKNOWN if unknown.
     */
    Enum fromFIFF( WLFiffLib::bem_surf_type_t bem );

    /**
     * Converts a BND bem name to a WLEBemType enum.
     *
     * \param unit FIFF unitm value
     * \return WLEBemType::Enum or WLEBemType::UNKNOWN if unknown.
     */
    Enum fromBND( std::string bemName );

} /* namespace WLEBemType */

inline std::ostream& operator<<( std::ostream &strm, const WLEBemType::Enum& obj )
{
    strm << WLEBemType::name( obj );
    return strm;
}

#endif  // WLEBEMTYPE_H_
