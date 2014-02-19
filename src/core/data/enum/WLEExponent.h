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

#ifndef WLEEXPONENT_H_
#define WLEEXPONENT_H_

#include <ostream>
#include <set>
#include <string>

#include "core/dataFormat/fiff/WLFiffUnitMultiplier.h"

/**
 * Enumeration for exponents, compatible with FIFF enum(unitm).
 *
 * \author pieloth
 */
namespace WLEExponent
{
    enum Enum
    {
        KILO = WLFiffLib::UnitMultiplier::K,
        BASE = WLFiffLib::UnitMultiplier::NONE,
        MILLI = WLFiffLib::UnitMultiplier::M,
        MICRO = WLFiffLib::UnitMultiplier::MU,
        NANO = WLFiffLib::UnitMultiplier::N,
        PICO = WLFiffLib::UnitMultiplier::P,
        FEMTO = WLFiffLib::UnitMultiplier::F,
        UNKNOWN = 128
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
     * \param val WLEExponent::Enum
     * \return A string.
     */
    std::string name( Enum val );

    /**
     * Returns the corresponding factor of the exponent, e.g. milli 1.0e-3
     *
     * \param val
     * \return decimal factor
     */
    double factor( Enum val );

    /**
     * Converts a FIFF unitm value to a WLEExponent enum.
     *
     * \param unit FIFF unitm value
     * \return WLEExponent::Enum or WLEExponent::BASE if unknown.
     */
    Enum fromFIFF( WLFiffLib::unitm_t unitm );

    std::ostream& operator<<( std::ostream &strm, const WLEExponent::Enum& obj );
} /* namespace WLEExponent */

inline std::ostream& WLEExponent::operator<<( std::ostream &strm, const WLEExponent::Enum& obj )
{
    strm << WLEExponent::name( obj );
    return strm;
}

#endif  // WLEEXPONENT_H_
