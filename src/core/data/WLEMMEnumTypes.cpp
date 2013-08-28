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

#include <string>
#include <vector>

#include <core/common/WAssert.h>

#include "WLEMMEnumTypes.h"

std::vector< LaBP::WEExponent::Enum > LaBP::WEExponent::values()
{
    std::vector< LaBP::WEExponent::Enum > exponents;
//    exponents.push_back( LaBP::WEExponent::KILO );
    exponents.push_back( LaBP::WEExponent::BASE );
    exponents.push_back( LaBP::WEExponent::MILLI );
    exponents.push_back( LaBP::WEExponent::MICRO );
    exponents.push_back( LaBP::WEExponent::NANO );
    exponents.push_back( LaBP::WEExponent::PICO );
    exponents.push_back( LaBP::WEExponent::FEMTO );
    return exponents;
}

std::string LaBP::WEExponent::name( LaBP::WEExponent::Enum val )
{
    switch( val )
    {
        case LaBP::WEExponent::KILO:
            return "10^3";
        case LaBP::WEExponent::BASE:
            return "1";
        case LaBP::WEExponent::MILLI:
            return "10^-3";
        case LaBP::WEExponent::MICRO:
            return "10^-6";
        case LaBP::WEExponent::NANO:
            return "10^-9";
        case LaBP::WEExponent::PICO:
            return "10^-12";
        case LaBP::WEExponent::FEMTO:
            return "10^-15";
        default:
            WAssert( false, "Unknown WEExponent!" );
            return UNDEFINED;
    }
}

double LaBP::WEExponent::factor( LaBP::WEExponent::Enum val )
{
    switch( val )
    {
        case LaBP::WEExponent::KILO:
            return 1.0e+3;
        case LaBP::WEExponent::BASE:
            return 1.0;
        case LaBP::WEExponent::MILLI:
            return 1.0e-3;
        case LaBP::WEExponent::MICRO:
            return 1.0e-6;
        case LaBP::WEExponent::NANO:
            return 1.0e-9;
        case LaBP::WEExponent::PICO:
            return 1.0e-12;
        case LaBP::WEExponent::FEMTO:
            return 1.0e-15;
        default:
            WAssert( false, "Unknown WEExponent!" );
            return 0.0;
    }
}

std::vector< LaBP::WEModalityType::Enum > LaBP::WEModalityType::values()
{
    std::vector< LaBP::WEModalityType::Enum > modalities;
    modalities.push_back( LaBP::WEModalityType::EEG );
    modalities.push_back( LaBP::WEModalityType::ECG );
    modalities.push_back( LaBP::WEModalityType::EOG );
    modalities.push_back( LaBP::WEModalityType::MEG );
    modalities.push_back( LaBP::WEModalityType::PCA );
    modalities.push_back( LaBP::WEModalityType::SOURCE );
    return modalities;
}

std::string LaBP::WEModalityType::name( LaBP::WEModalityType::Enum val )
{
    switch( val )
    {
        case LaBP::WEModalityType::EEG:
            return "EEG";
        case LaBP::WEModalityType::ECG:
            return "ECG";
        case LaBP::WEModalityType::EOG:
            return "EOG";
        case LaBP::WEModalityType::MEG:
            return "MEG";
        case LaBP::WEModalityType::PCA:
            return "PCA";
        case LaBP::WEModalityType::SOURCE:
                    return "Source";
        default:
            WAssert( false, "Unknown WEModalityType!" );
            return UNDEFINED;
    }

}

std::string LaBP::WEModalityType::description( LaBP::WEModalityType::Enum val )
{
    switch( val )
    {
        case LaBP::WEModalityType::EEG:
            return "EEG measurement";
        case LaBP::WEModalityType::ECG:
            return "ECG measurement";
        case LaBP::WEModalityType::EOG:
            return "EOG measurement";
        case LaBP::WEModalityType::MEG:
            return "MEG measurement";
        case LaBP::WEModalityType::PCA:
            return "PCA measurement";
        case LaBP::WEModalityType::SOURCE:
                    return "Source localization";
        default:
            WAssert( false, "Unknown WEModalityType!" );
            return UNDEFINED;
    }

}

std::vector< LaBP::WEPolarityType::Enum > LaBP::WEPolarityType::values()
{
    std::vector< LaBP::WEPolarityType::Enum > options;
    options.push_back( LaBP::WEPolarityType::BIPOLAR );
    options.push_back( LaBP::WEPolarityType::UNIPOLAR );
    return options;
}

std::vector< LaBP::WEGeneralCoilType::Enum > LaBP::WEGeneralCoilType::values()
{
    std::vector< LaBP::WEGeneralCoilType::Enum > modalities;
    modalities.push_back( LaBP::WEGeneralCoilType::MAGNETOMETER );
    modalities.push_back( LaBP::WEGeneralCoilType::GRADIOMETER );
    return modalities;
}

std::vector< LaBP::WESpecificCoilType::Enum > LaBP::WESpecificCoilType::values()
{
    std::vector< LaBP::WESpecificCoilType::Enum > modalities;
    return modalities;
}

std::vector< LaBP::WEUnit::Enum > LaBP::WEUnit::values()
{
    std::vector< LaBP::WEUnit::Enum > values;
    values.push_back( LaBP::WEUnit::SIEMENS_PER_METER );
    values.push_back( LaBP::WEUnit::METER );
    values.push_back( LaBP::WEUnit::VOLT );
    values.push_back( LaBP::WEUnit::TESLA );
    values.push_back( LaBP::WEUnit::TESLA_PER_METER );
    values.push_back( LaBP::WEUnit::UNKNOWN_UNIT );
    values.push_back( LaBP::WEUnit::UNITLESS );
    return values;
}

std::vector< LaBP::WECoordSystemName::Enum > LaBP::WECoordSystemName::values()
{
    std::vector< LaBP::WECoordSystemName::Enum > modalities;
    modalities.push_back( LaBP::WECoordSystemName::HEAD );
    modalities.push_back( LaBP::WECoordSystemName::DEVICE );
    modalities.push_back( LaBP::WECoordSystemName::AC_PC );
    return modalities;
}

std::vector< LaBP::WESex::Enum > LaBP::WESex::values()
{
    std::vector< LaBP::WESex::Enum > modalities;
    modalities.push_back( LaBP::WESex::MALE );
    modalities.push_back( LaBP::WESex::FEMALE );
    modalities.push_back( LaBP::WESex::OTHER );
    return modalities;
}

std::vector< LaBP::WEHand::Enum > LaBP::WEHand::values()
{
    std::vector< LaBP::WEHand::Enum > modalities;
    modalities.push_back( LaBP::WEHand::RIGHT );
    modalities.push_back( LaBP::WEHand::LEFT );
    modalities.push_back( LaBP::WEHand::BOTH );
    return modalities;
}

std::vector< LaBP::WEBemType::Enum > LaBP::WEBemType::values()
{
    std::vector< LaBP::WEBemType::Enum > values;
    values.push_back( LaBP::WEBemType::BRAIN );
    values.push_back( LaBP::WEBemType::SKULL );
    values.push_back( LaBP::WEBemType::SKIN );
    values.push_back( LaBP::WEBemType::INNER_SKIN );
    values.push_back( LaBP::WEBemType::OUTER_SKIN );
    values.push_back( LaBP::WEBemType::INNER_SKULL );
    values.push_back( LaBP::WEBemType::OUTER_SKULL );
    return values;
}

std::string LaBP::WEBemType::name( LaBP::WEBemType::Enum val )
{
    switch( val )
    {
        case LaBP::WEBemType::BRAIN:
            return "Brain";
        case LaBP::WEBemType::SKULL:
            return "Skull";
        case LaBP::WEBemType::SKIN:
            return "Skin";
        case LaBP::WEBemType::INNER_SKIN:
            return "inner_skin";
        case LaBP::WEBemType::OUTER_SKIN:
            return "outer_skin";
        case LaBP::WEBemType::INNER_SKULL:
            return "inner_skull";
        case LaBP::WEBemType::OUTER_SKULL:
            return "outer_skull";
        default:
            WAssert( false, "Unknown WEBemType!" );
            return UNDEFINED;
    }
}
