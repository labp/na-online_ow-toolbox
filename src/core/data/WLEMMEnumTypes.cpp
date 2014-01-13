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

using namespace LaBP;

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
