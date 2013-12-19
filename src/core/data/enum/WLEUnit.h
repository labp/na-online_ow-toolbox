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

#ifndef WLEUNIT_H_
#define WLEUNIT_H_

#include <set>

#include "core/fileFormat/fiff/WLFiffUnit.h"

/**
 * Enumeration for units, compatible with FIFF enum(units).
 *
 * \author pieloth
 */
namespace WLEUnit
{
    enum Enum
    {
        NONE = WLFiffLib::Unit::NONE,         //!< NONE
        UNITLESS = WLFiffLib::Unit::UNITLESS, //!< UNITLESS
        METER = WLFiffLib::Unit::M,           //!< METER
        VOLT = WLFiffLib::Unit::V,            //!< VOLT
        TESLA = WLFiffLib::Unit::T,           //!< TESLA
        TESLA_PER_METER = WLFiffLib::Unit::TM,           //!< TESLA_PER_METER
        SIEMENS_PER_METER = 1000              //!< SIEMENS_PER_METER
    };

    typedef std::set< Enum > ContainerT;

    /**
     * Gets all enum values.
     *
     * \return Container with all enum values.
     */
    ContainerT values();

    /**
     * Converts a FIFF unit value to a WLEUnit enum.
     *
     * \param unit FIFF unit value
     * \return WLEUnit::Enum or WLEUnit::NONE if unknown.
     */
    Enum convertFIFF( WLFiffLib::unit_t unit );
}

#endif  // WLEUNIT_H_
