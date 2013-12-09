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
    std::set< Enum > values();
    std::string name( Enum val );
    std::string description( Enum val );

    std::set< Enum > valuesDevice();
    bool isDevice( Enum );

    std::set< Enum > valuesMEG();
    bool isMEG( Enum );

    std::set< Enum > valuesComputed();
    bool isComputed( Enum );

    std::set< Enum > valuesLocalizeable();
    bool isLocalizeable( Enum );
}

inline bool WLEModality::isDevice( WLEModality::Enum val )
{
    return val == WLEModality::EEG || val == WLEModality::MEG || val == WLEModality::ECG || val == WLEModality::EOG;
}

inline bool WLEModality::isMEG( WLEModality::Enum val )
{
    return val == WLEModality::MEG || val == WLEModality::MEG_MAG || val == WLEModality::MEG_GRAD
                    || val == WLEModality::MEG_GRAD_MERGED;
}

inline bool WLEModality::isComputed( WLEModality::Enum val )
{
    return val == WLEModality::SOURCE || val == WLEModality::PCA;
}

inline bool WLEModality::isLocalizeable( WLEModality::Enum val )
{
    return val == WLEModality::EEG || val == WLEModality::MEG;
}

#endif  // WLEMODALITY_H_
