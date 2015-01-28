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

#ifndef WLEUNIT_H_
#define WLEUNIT_H_

#include <ostream>
#include <set>
#include <string>

#include "core/dataFormat/fiff/WLFiffUnit.h"

/**
 * Enumeration for units, compatible with FIFF enum(units).
 *
 * \author pieloth
 * \ingroup data
 */
namespace WLEUnit
{
    enum Enum
    {
        UNKNOWN = WLFiffLib::Unit::NONE,         //!< none/Unknown
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
     * Gets the name of the enum value.
     *
     * \param val WLEUnit::Enum
     * \return A string.
     */
    std::string name( Enum val );

    /**
     * Converts a FIFF unit value to a WLEUnit enum.
     *
     * \param unit FIFF unit value
     * \return WLEUnit::Enum or WLEUnit::NONE if unknown.
     */
    Enum fromFIFF( WLFiffLib::unit_t unit );

    std::ostream& operator<<( std::ostream &strm, const WLEUnit::Enum& obj );
}

inline std::ostream& WLEUnit::operator<<( std::ostream &strm, const WLEUnit::Enum& obj )
{
    strm << WLEUnit::name( obj );
    return strm;
}

#endif  // WLEUNIT_H_
