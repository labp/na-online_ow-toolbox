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

#ifndef WLEMODALITY_H_
#define WLEMODALITY_H_

#include <ostream>
#include <set>
#include <string>

/**
 * Enumeration for modalities.
 *
 * \author pieloth
 */
namespace WLEModality
{
    enum Enum
    {
        EEG = 0, ECG = 1, MEG = 2, EOG = 3, PCA = 4, SOURCE = 5, MEG_MAG = 6, MEG_GRAD = 7, MEG_GRAD_MERGED = 8, UNKNOWN = -1
    };

    typedef std::set< Enum > ContainerT;

    ContainerT values();
    std::string name( Enum val );
    std::string description( Enum val );

    ContainerT valuesDevice();
    bool isDevice( Enum );

    ContainerT valuesMEG();
    bool isMEG( Enum );

    ContainerT valuesMEGCoil();
    bool isMEGCoil( Enum );

    ContainerT valuesComputed();
    bool isComputed( Enum );

    ContainerT valuesLocalizeable();
    bool isLocalizeable( Enum );
}

inline bool WLEModality::isDevice( WLEModality::Enum val )
{
    return val == WLEModality::EEG || val == WLEModality::MEG || val == WLEModality::ECG || val == WLEModality::EOG;
}

inline bool WLEModality::isMEG( WLEModality::Enum val )
{
    return val == WLEModality::MEG || isMEGCoil( val );
}

inline bool WLEModality::isMEGCoil( WLEModality::Enum val )
{
    return val == WLEModality::MEG_MAG || val == WLEModality::MEG_GRAD || val == WLEModality::MEG_GRAD_MERGED;
}

inline bool WLEModality::isComputed( WLEModality::Enum val )
{
    return val == WLEModality::SOURCE || val == WLEModality::PCA;
}

inline bool WLEModality::isLocalizeable( WLEModality::Enum val )
{
    return val == WLEModality::EEG || val == WLEModality::MEG;
}

inline std::ostream& operator<<( std::ostream &strm, const WLEModality::Enum& obj )
{
    strm << WLEModality::name( obj );
    return strm;
}

#endif  // WLEMODALITY_H_
