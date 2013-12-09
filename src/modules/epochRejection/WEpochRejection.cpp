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

/**
 * Constructor
 */
WEpochRejection::WEpochRejection()
{
    initRejection();
}

/**
 * Destructor
 */
WEpochRejection::~WEpochRejection()
{

}

/**
 * Method to reset the thresholds.
 */
void WEpochRejection::initRejection()
{
    m_eegLevel = 0;
    m_eogLevel = 0;
    m_megGrad = 0;
    m_megMag = 0;
    m_rejCount = 0;
}

/**
 * Method to initialize the thresholds
 */
void WEpochRejection::setLevels( double eegLevel, double eogLevel, double megGrad, double megMag )
{
    m_eegLevel = eegLevel;
    m_eogLevel = eogLevel;
    m_megGrad = megGrad;
    m_megMag = megMag;
}

/**
 * Method to process the rejection on the data.
 */
bool WEpochRejection::getRejection( const WLEMMeasurement::SPtr emm )
{
    WLTimeProfiler tp( "WEpochRejection", "getRejection" );
    wlog::debug( CLASS ) << "starting rejection";

    m_rejCount = 0;

    WLEMData::SPtr modality;

    for( size_t mod = 0; mod < emm->getModalityCount(); ++mod ) // for all modalities
    {
        // get modality
        modality = emm->getModality( mod );
        const size_t channels = modality->getData().rows();
        size_t rejections = 0;

        // if wrong modality, next
        if( !validModality( modality->getModalityType() ) )
        {
            wlog::debug( CLASS ) << "invalid modality";
            continue;
        }

        WLEMData::SampleT max = modality->getData().rowwise().maxCoeff();
        WLEMData::SampleT min = modality->getData().rowwise().minCoeff();
        WLEMData::SampleT diff = max - min;

        // definition of the threshold to use by the modality
        switch( modality->getModalityType() )
        {
            case WLEModality::EEG:
                // compare the difference with the given level value
                if( diff.maxCoeff() > m_eegLevel )
                {
                    ++rejections; // counts the rejected for each modality
                }
                break;
            case WLEModality::EOG:
                if( diff.maxCoeff() > m_eogLevel )
                {
                    ++rejections; // counts the rejected for each modality
                }
                break;
            case WLEModality::MEG:

                for( size_t chan = 0; chan < channels; chan++ )
                {
                    if( ( chan % 3 ) == 0 ) // magnetometer
                    {
                        if( diff( chan ) > m_megMag )
                        {
                            ++rejections;
                        }

                    }
                    else // gradiometer
                    {
                        if( diff( chan ) > m_megGrad )
                        {
                            ++rejections;
                        }
                    }

                }

                break;
            default:
                ; // skip the channel
        }

        // if least one channel has to reject, reject the whole input
        if( rejections > 0 )
        {
            wlog::debug( CLASS ) << "Epochs rejected for " << modality->getModalityType() << ": " << rejections;
            ++m_rejCount;
        }
    }

    return m_rejCount > 0;
}

/**
 * Returns the number of rejections.
 */
size_t WEpochRejection::getCount()
{
    return m_rejCount;
}

/**
 * Method to separate valid modalities from invalid modalities.
 * It returns false, if the modality has to skip else true.
 */
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
        default:
            ;
    }

    return rc;
}
