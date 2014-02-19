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

#ifndef WLEMEGGENERALCOILTYPE_H_
#define WLEMEGGENERALCOILTYPE_H_

#include <ostream>
#include <set>
#include <string>

namespace WLEMEGGeneralCoilType
{
    enum Enum
    {
        MAGNETOMETER, GRADIOMETER
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
     * \param val WLEMEGGeneralCoilType::Enum
     * \return A string.
     */
    std::string name( Enum val );

    std::ostream& operator<<( std::ostream &strm, const WLEMEGGeneralCoilType::Enum& obj );
}

inline std::ostream& WLEMEGGeneralCoilType::operator<<( std::ostream &strm, const WLEMEGGeneralCoilType::Enum& obj )
{
    strm << WLEMEGGeneralCoilType::name( obj );
    return strm;
}

#endif  // WLEMEGGENERALCOILTYPE_H_
