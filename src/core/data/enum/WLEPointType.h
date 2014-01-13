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

#ifndef WLEPOINTTYPE_H_
#define WLEPOINTTYPE_H_

#include <ostream>
#include <set>
#include <string>

#include "core/dataFormat/fiff/WLFiffPointType.h"

/**
 * Enumeration for point types, compatible with FIFF enum(point).
 *
 * \author pieloth
 */
namespace WLEPointType
{
    enum Enum
    {
        UNKNOWN = -1,
        CARDINAL = WLFiffLib::PointType::CARDINAL,
        HPI = WLFiffLib::PointType::HPI,
        EEG_ECG = WLFiffLib::PointType::EEG, // WLFiffLib::PointType::ECG
        EXTRA = WLFiffLib::PointType::EXTRA,
        HEAD_SURFACE = WLFiffLib::PointType::HEAD_SURFACE
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
     * \param val WLEPointType::Enum
     * \return A string.
     */
    std::string name( Enum val );

    /**
     * Converts a FIFF point value to a WLEPointType enum.
     *
     * \param val FIFF point value
     * \return WLEPointType::Enum or WLEPointType::UNKNOWN if unknown.
     */
    Enum fromFIFF( WLFiffLib::point_type_t val );
}

inline std::ostream& operator<<( std::ostream &strm, const WLEPointType::Enum& obj )
{
    strm << WLEPointType::name( obj );
    return strm;
}

#endif  // WLEPOINTTYPE_H_
