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

#ifndef WLECOILTYPE_H_
#define WLECOILTYPE_H_

#include <ostream>
#include <set>
#include <string>

#include "core/dataFormat/fiff/WLFiffCoilType.h"

/**
 * Enumeration for coil types, compatible with FIFF enum(coil).
 *
 * \author pieloth
 */
namespace WLECoilType
{
    enum Enum
    {
        NONE = WLFiffLib::CoilType::NONE,
        EEG = WLFiffLib::CoilType::EEG,
        EEG_BIPOLAR = WLFiffLib::CoilType::EEG_BIPOLAR,
        VV_PLANAR_W = WLFiffLib::CoilType::VV_PLANAR_W,
        VV_PLANAR_T1 = WLFiffLib::CoilType::VV_PLANAR_T1,
        VV_PLANAR_T2 = WLFiffLib::CoilType::VV_PLANAR_T2,
        VV_PLANAR_T3 = WLFiffLib::CoilType::VV_PLANAR_T3,
        VV_MAG_W = WLFiffLib::CoilType::VV_MAG_W,
        VV_MAG_T1 = WLFiffLib::CoilType::VV_MAG_T1,
        VV_MAG_T2 = WLFiffLib::CoilType::VV_MAG_T2,
        VV_MAG_T3 = WLFiffLib::CoilType::VV_MAG_T3
    };

    typedef std::set< Enum > ContainerT;

    /**
     * Gets all enum values.
     *
     * \return Container with all enum values.
     */
    ContainerT values();

    /**
     * Gets enum values relating to EEG.
     *
     * \return Container with enum values for EEG.
     */
    ContainerT valuesEEG();

    /**
     * Checks if coil is an EEG coil.
     *
     * @param val coil to check
     * @return true if coil is an EEG coil
     */
    bool isEEG( Enum val );

    /**
     * Gets enum values relating to MEG.
     *
     * \return Container with enum values for MEG.
     */
    ContainerT valuesMEG();

    /**
     * Checks if coil is a MEG coil.
     *
     * @param val coil to check
     * @return true if coil is a MEG coil
     */
    bool isMEG( Enum val );

    /**
     * Gets enum values relating to magnetometer.
     *
     * \return Container with enum values for magnetometer.
     */
    ContainerT valuesMagnetometer();

    /**
     * Checks if coil is a magnetometer coil.
     *
     * @param val coil to check
     * @return true if coil is a magnetometer coil
     */
    bool isMagnetometer( Enum val );

    /**
     * Gets enum values relating to gradiometer.
     *
     * \return Container with enum values for gradiometer.
     */
    ContainerT valuesGradiometer();

    /**
     * Checks if coil is a gradiometer coil.
     *
     * @param val coil to check
     * @return true if coil is a gradiometer coil
     */
    bool isGradiometer( Enum val );

    /**
     * Gets the name of the enum value.
     *
     * \param val WLECoilType::Enum
     * \return A string.
     */
    std::string name( Enum val );

    /**
     * Converts a FIFF coil value to a WLECoilType enum.
     *
     * \param val FIFF coil value
     * \return WLECoilType::Enum or WLECoilType::NONE if unknown.
     */
    Enum fromFIFF( WLFiffLib::coil_type_t val );

    std::ostream& operator<<( std::ostream &strm, const WLECoilType::Enum& obj );
}

inline bool WLECoilType::isEEG( WLECoilType::Enum val )
{
    return WLECoilType::EEG == val || WLECoilType::EEG_BIPOLAR == val;
}

inline bool WLECoilType::isMEG( WLECoilType::Enum val )
{
    return WLECoilType::VV_MAG_W == val || WLECoilType::VV_MAG_T1 == val || WLECoilType::VV_MAG_T2 == val
                    || WLECoilType::VV_MAG_T3 == val || WLECoilType::VV_PLANAR_W == val || WLECoilType::VV_PLANAR_T1 == val
                    || WLECoilType::VV_PLANAR_T2 == val || WLECoilType::VV_PLANAR_T3 == val;
}

inline bool WLECoilType::isMagnetometer( WLECoilType::Enum val )
{
    return WLECoilType::VV_MAG_W == val || WLECoilType::VV_MAG_T1 == val || WLECoilType::VV_MAG_T2 == val
                    || WLECoilType::VV_MAG_T3 == val;
}

inline bool WLECoilType::isGradiometer( WLECoilType::Enum val )
{
    return WLECoilType::VV_PLANAR_W == val || WLECoilType::VV_PLANAR_T1 == val || WLECoilType::VV_PLANAR_T2 == val
                    || WLECoilType::VV_PLANAR_T3 == val;
}

inline std::ostream& WLECoilType::operator<<( std::ostream &strm, const WLECoilType::Enum& obj )
{
    strm << WLECoilType::name( obj );
    return strm;
}

#endif  // WLECOILTYPE_H_
