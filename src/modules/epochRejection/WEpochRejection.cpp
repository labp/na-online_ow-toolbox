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

#include <core/common/WLogger.h>

#include "core/data/enum/WLEModality.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WEpochRejection.h"

const std::string WEpochRejection::CLASS = "WEpochRejection";

WEpochRejection::WEpochRejection()
{
    m_eegThreshold = 0;
    m_eogThreshold = 0;
    m_megGradThreshold = 0;
    m_megMagThreshold = 0;
    m_rejCount = 0;
}

WEpochRejection::~WEpochRejection()
{

}

void WEpochRejection::setThresholds( double eegLevel, double eogLevel, double megGrad, double megMag )
{
    m_eegThreshold = eegLevel;
    m_eogThreshold = eogLevel;
    m_megGradThreshold = megGrad;
    m_megMagThreshold = megMag;
}

size_t WEpochRejection::getCount() const
{
    return m_rejCount;
}

bool WEpochRejection::validModality( WLEModality::Enum modalityType )
{
    bool rc = false;

    switch( modalityType )
    {
        case WLEModality::EEG:
            rc = true;
            break;
        case WLEModality::EOG:
            rc = true;
            break;
        case WLEModality::MEG:
            rc = true;
            break;
        case WLEModality::MEG_GRAD:
            rc = true;
            break;
        case WLEModality::MEG_MAG:
            rc = true;
            break;
        case WLEModality::MEG_GRAD_MERGED:
            rc = true;
            break;
        default:
            ;
    }

    return rc;
}

double WEpochRejection::getThreshold( WLEModality::Enum modalityType, size_t channelNo )
{
    switch( modalityType )
    {
        case WLEModality::EEG:
            return m_eegThreshold;
        case WLEModality::EOG:
            return m_eogThreshold;
        case WLEModality::MEG:
            if( ( channelNo % 3 ) == 0 ) // magnetometer
            {
                return m_megMagThreshold;
            }
            else
            {
                return m_megGradThreshold; // gradiometer
            }
        case WLEModality::MEG_GRAD:
            return m_megGradThreshold;
        case WLEModality::MEG_MAG:
            return m_megMagThreshold;
        case WLEModality::MEG_GRAD_MERGED:
            if( ( channelNo % 3 ) == 0 ) // magnetometer
            {
                return m_megMagThreshold;
            }
            else
            {
                return m_megGradThreshold; // gradiometer
            }
        default:
            return 0;
    }
}
