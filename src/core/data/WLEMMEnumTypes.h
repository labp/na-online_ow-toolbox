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

#include <set>
#include <string>
#include <vector>

#include <core/common/WDefines.h>

/**
 * TODO(kaehler): Comments
 */
namespace LaBP
{

    const std::string UNDEFINED = "UNDEFINED";

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
}

#endif  // WDATASETEMMENUMTYPES_H
