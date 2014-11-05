//---------------------------------------------------------------------------
//
// Project: NA-Online ( http://www.labp.htwk-leipzig.de )
//
// Copyright 2010 Laboratory for Biosignal Processing, HTWK Leipzig, Germany
//
// This file is part of NA-Online.
//
// NA-Online is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// NA-Online is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with NA-Online. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#include <string>

#include <boost/foreach.hpp>

#include <core/common/WLogger.h>

#include "core/data/enum/WLEModality.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WEpochRejection.h"

const std::string WEpochRejection::CLASS = "WEpochRejection";

WEpochRejection::WEpochRejection()
{
    m_rejCount = 0;

    m_thresholdList.reset( new WThreshold::WThreshold_List() );
}

WEpochRejection::~WEpochRejection()
{
}

WThreshold::WThreshold_List_SPtr WEpochRejection::getThresholds()
{
    return m_thresholdList;
}

void WEpochRejection::setThresholds( WThreshold::WThreshold_List_SPtr thresholdList )
{
    m_thresholdList = thresholdList;
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
            {
            }
    }

    return rc;
}

double WEpochRejection::getThreshold( WLEModality::Enum modalityType, size_t channelNo )
{
    if( WLEModality::isMEG( modalityType ) )
    {
        if( ( channelNo % 3 ) == 0 ) // magnetometer
        {
            modalityType = WLEModality::MEG_MAG;
        }
        else
        {
            modalityType = WLEModality::MEG_GRAD; // gradiometer
        }
    }

    return getThreshold( modalityType );
}

double WEpochRejection::getThreshold( WLEModality::Enum modalityType )
{
    BOOST_FOREACH( WThreshold threshold, *m_thresholdList.get() )
    {
        if( threshold.getModaliyType() == modalityType )
        {
            return threshold.getValue();
        }
    }

    return 0;
}

void WEpochRejection::showThresholds()
{
    BOOST_FOREACH( WThreshold threshold, *m_thresholdList.get() )
    {
        wlog::debug( CLASS ) << WLEModality::name( threshold.getModaliyType() ) << ": " << threshold.getValue();
    }
}
