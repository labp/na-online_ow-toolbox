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

#ifndef WDATASETEMMENUMTYPES_H
#define WDATASETEMMENUMTYPES_H

#include <string>
#include <vector>

/**
 * TODO(kaehler): Comments
 */
namespace LaBP
{

    const std::string UNDEFINED = "UNDEFINED";

    /**
     * TODO(kaehler): Comments
     */
    namespace WEPolarityType
    {
        enum Enum
        {
            BIPOLAR, UNIPOLAR
        };
        std::vector< Enum > values();
    }
    /**
     * TODO(kaehler): Comments
     */
    namespace WEGeneralCoilType
    {

        enum Enum
        {
            MAGNETOMETER, GRADIOMETER
        };
        std::vector< Enum > values();
    }

    /**
     * TODO(kaehler): Comments
     */
    namespace WESpecificCoilType
    {
        enum Enum
        {
        // TODO(kaehler): Tabelle übertragen
        };
        std::vector< Enum > values();
    }
    /**
     * TODO(kaehler): Comments
     */
    namespace WEUnit
    {
        enum Enum
        {
            SIEMENS_PER_METER, METER, VOLT, TESLA, TESLA_PER_METER, UNKNOWN_UNIT, UNITLESS
        };
        std::vector< Enum > values();
    }

    /**
     * TODO(kaehler): Comments
     */
    namespace WEExponent
    {
        enum Enum
        {
            KILO = 3,
            BASE = 0,
            MILLI = -3,
            MICRO = -6,
            NANO = -9,
            PICO = -12,
            FEMTO = -15
        };
        std::vector< Enum > values();
        std::string name( Enum val );
        double factor( Enum val);

    }

    /**
     * TODO(kaehler): Comments
     */
    namespace WECoordSystemName
    {
        enum Enum
        {
            HEAD, DEVICE, AC_PC
        };
        std::vector< Enum > values();
    }

    /**
     * TODO(kaehler): Comments
     */

    namespace WEModalityType
    {
        enum Enum
        {
            EEG = 0, ECG = 1, MEG = 2, EOG = 3, PCA = 4, SOURCE = 5
        };
        std::vector< Enum > values();
        std::string name( Enum val );
        std::string description( Enum val );
    }

    /**
     * TODO(kaehler): Comments
     */
    namespace WESex
    {
        enum Enum
        {
            MALE, FEMALE, OTHER
        };
        std::vector< Enum > values();
    }
    /**
     * TODO(kaehler): Comments
     */
    namespace WEHand
    {
        enum Enum
        {
            RIGHT, LEFT, BOTH
        };
        std::vector< Enum > values();
    }

    namespace WEBemType
    {
        enum Enum
        {
            BRAIN, SKULL, SKIN, INNER_SKIN, OUTER_SKIN, INNER_SKULL, OUTER_SKULL, UNKNOWN, UNKNOWN2
        };
        std::vector< Enum > values();
        std::string name( Enum val );
    }

}

#endif  // WDATASETEMMENUMTYPES_H
