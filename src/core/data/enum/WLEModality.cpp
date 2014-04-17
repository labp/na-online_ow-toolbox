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

#include <stdarg.h>

#include <core/common/WAssert.h>

#include "WLEModality.h"

WLEModality::ContainerT WLEModality::values()
{
    ContainerT modalities;
    modalities.insert( WLEModality::EEG );
    modalities.insert( WLEModality::ECG );
    modalities.insert( WLEModality::EOG );
    modalities.insert( WLEModality::MEG );
    modalities.insert( WLEModality::PCA );
    modalities.insert( WLEModality::SOURCE );
    modalities.insert( WLEModality::MEG_MAG );
    modalities.insert( WLEModality::MEG_GRAD );
    modalities.insert( WLEModality::MEG_GRAD_MERGED );
    return modalities;
}

std::string WLEModality::name( WLEModality::Enum val )
{
    switch( val )
    {
        case WLEModality::EEG:
            return "EEG";
        case WLEModality::ECG:
            return "ECG";
        case WLEModality::EOG:
            return "EOG";
        case WLEModality::MEG:
            return "MEG";
        case WLEModality::PCA:
            return "PCA";
        case WLEModality::SOURCE:
            return "Source";
        case WLEModality::MEG_MAG:
            return "MEG (mag.)";
        case WLEModality::MEG_GRAD:
            return "MEG (grad.)";
        case WLEModality::MEG_GRAD_MERGED:
            return "MEG (grad2)";
        case WLEModality::UNKNOWN:
            return "UNKNOWN";
        default:
            WAssert( false, "Unknown WLEModality!" );
            return WLEModality::name( WLEModality::UNKNOWN );
    }
}

std::string WLEModality::description( WLEModality::Enum val )
{
    switch( val )
    {
        case WLEModality::EEG:
            return "EEG measurement";
        case WLEModality::ECG:
            return "ECG measurement";
        case WLEModality::EOG:
            return "EOG measurement";
        case WLEModality::MEG:
            return "MEG measurement";
        case WLEModality::PCA:
            return "PCA measurement";
        case WLEModality::SOURCE:
            return "Source localization";
        case WLEModality::MEG_MAG:
            return "MEG measurement (magnetometer)";
        case WLEModality::MEG_GRAD:
            return "MEG measurement (gradiometer)";
        case WLEModality::MEG_GRAD_MERGED:
            return "MEG measurement (gradiometer merged)";
        case WLEModality::UNKNOWN:
            return "UNKNOWN modality type!";
        default:
            WAssert( false, "Unknown WLEModality!" );
            return WLEModality::name( WLEModality::UNKNOWN );
    }
}

WLEModality::ContainerT WLEModality::valuesDevice()
{
    ContainerT modalities;
    modalities.insert( WLEModality::EEG );
    modalities.insert( WLEModality::ECG );
    modalities.insert( WLEModality::EOG );
    modalities.insert( WLEModality::MEG );
    return modalities;
}

WLEModality::ContainerT WLEModality::valuesMEG()
{
    ContainerT modalities;
    modalities.insert( WLEModality::MEG );
    modalities.insert( WLEModality::MEG_MAG );
    modalities.insert( WLEModality::MEG_GRAD );
    modalities.insert( WLEModality::MEG_GRAD_MERGED );
    return modalities;
}

WLEModality::ContainerT WLEModality::valuesMEGCoil()
{
    ContainerT modalities;
    modalities.insert( WLEModality::MEG_MAG );
    modalities.insert( WLEModality::MEG_GRAD );
    modalities.insert( WLEModality::MEG_GRAD_MERGED );
    return modalities;
}

WLEModality::ContainerT WLEModality::valuesComputed()
{
    ContainerT modalities;
    modalities.insert( WLEModality::PCA );
    modalities.insert( WLEModality::SOURCE );
    return modalities;
}

WLEModality::ContainerT WLEModality::valuesLocalizeable()
{
    ContainerT modalities;
    modalities.insert( WLEModality::EEG );
    modalities.insert( WLEModality::MEG );
    return modalities;
}

WLEModality::Enum WLEModality::fromFiffType( int kind )
{
    switch( kind )
    {
        case 1:
            return WLEModality::MEG;
        case 2:
            return WLEModality::EEG;
        case 202:
            return WLEModality::EOG;
        case 402:
            return WLEModality::ECG;
        default:
            return WLEModality::UNKNOWN;
    }
}
