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

#include "WEpochRejection.h"

const std::string WEpochRejection::CLASS = "WEpochRejection";

WEpochRejection::WEpochRejection()
{
    initRejection();
}

WEpochRejection::~WEpochRejection()
{

}

void WEpochRejection::initRejection()
{
    m_eegLevel = 0;
    m_eogLevel = 0;
    m_megGrad = 0;
    m_megMag = 0;
    m_rejCount = 0;
}

void WEpochRejection::setThresholds( double eegLevel, double eogLevel, double megGrad, double megMag )
{
    m_eegLevel = eegLevel;
    m_eogLevel = eogLevel;
    m_megGrad = megGrad;
    m_megMag = megMag;
}

size_t WEpochRejection::getCount() const
{
    return m_rejCount;
}

bool WEpochRejection::validModality( LaBP::WEModalityType::Enum modalityType )
{
    bool rc = false;

    switch( modalityType )
    {
        case LaBP::WEModalityType::EEG:
            rc = true;
            break;
        case LaBP::WEModalityType::EOG:
            rc = true;
            break;
        case LaBP::WEModalityType::MEG:
            rc = true;
            break;
        default:
            ;
    }

    return rc;
}
