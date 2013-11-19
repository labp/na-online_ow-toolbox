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

#include "core/common/WLogger.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WEpochRejection.h"
#include "WEpochRejectionSingle.h"
#include "WBadChannelManager.h"

const std::string WEpochRejectionSingle::CLASS = "WEpochRejectionSingle";

/**
 * Constructor
 */
WEpochRejectionSingle::WEpochRejectionSingle() :
                WEpochRejection::WEpochRejection()
{
}

/**
 * Destructor
 */
WEpochRejectionSingle::~WEpochRejectionSingle()
{
}

/**
 * Method to process the rejection on the data.
 */
bool WEpochRejectionSingle::getRejection( const WLEMMeasurement::SPtr emm )
{
    WLTimeProfiler tp( "WEpochRejectionSingle", "getRejection" );
    wlog::debug( CLASS ) << "starting single channel rejection";

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

        WLEMData::SampleT max = modality->getData().rowwise().maxCoeff(); // get the maximum for all channels
        WLEMData::SampleT min = modality->getData().rowwise().minCoeff(); // get the minimum for all channels
        WLEMData::SampleT diff = max - min; // calculate the difference for all channels

        // definition of the threshold to use by the modality
        switch( modality->getModalityType() )
        {
            case LaBP::WEModalityType::EEG:
                // compare the difference with the given level value

                for( size_t chan = 0; chan < channels; ++chan )
                {
                    if( diff( chan ) > m_eegLevel )
                    {
                        // collect the invalid channel
                        //WBadChannelManager::instance()->addElement( modality->getModalityType(), chan );

                        ++rejections; // counts the rejected for each modality
                    }
                }
                break;
            case LaBP::WEModalityType::EOG:
                for( size_t chan = 0; chan < channels; ++chan )
                {
                    if( diff( chan ) > m_eogLevel )
                    {
                        // collect the invalid channel
                        //WInvalidChannelManager::instance()->addChannel( modality->getModalityType(), chan );

                        ++rejections; // counts the rejected for each modality
                    }
                }
                break;
            case LaBP::WEModalityType::MEG:

                for( size_t chan = 0; chan < channels; ++chan )
                {
                    if( ( chan % 3 ) == 0 ) // magnetometer
                    {
                        if( diff( chan ) > m_megMag )
                        {
                            // collect the invalid channel
                            WBadChannelManager::instance()->addElement( modality->getModalityType(), chan );

                            ++rejections;
                        }

                    }
                    else // gradiometer
                    {
                        if( diff( chan ) > m_megGrad )
                        {
                            // collect the invalid channel
                            WBadChannelManager::instance()->addElement( modality->getModalityType(), chan );

                            ++rejections;
                        }
                    }

                }

                break;
            default:
                ; // skip the modality
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
