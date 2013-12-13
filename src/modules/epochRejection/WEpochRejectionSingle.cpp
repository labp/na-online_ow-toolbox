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
#include "core/data/emd/WLEMData.h"

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
    m_BadChannelUpdated = false;
}

/**
 * Destructor
 */
WEpochRejectionSingle::~WEpochRejectionSingle()
{
}

bool WEpochRejectionSingle::isBadChannelUpdated() const
{
    return m_BadChannelUpdated;
}

bool WEpochRejectionSingle::doRejection( const WLEMMeasurement::ConstSPtr emm )
{
    WLTimeProfiler tp( "WEpochRejectionSingle", "getRejection" );
    wlog::debug( CLASS ) << "starting single channel rejection";

    m_rejCount = 0;
    m_BadChannelUpdated = false;

    for( size_t mod = 0; mod < emm->getModalityCount(); ++mod ) // for all modalities
    {
        // get modality
        WLEMData::ConstSPtr modality = emm->getModality( mod );
        const size_t channels = modality->getData().rows();
        size_t rejections = 0;
        size_t channelNo = 0;
        double threshold;

        // if wrong modality, next
        if( !validModality( modality->getModalityType() ) )
        {
            wlog::debug( CLASS ) << "invalid modality";
            continue;
        }

        // iterate all channels
        for( size_t chan = 0; chan < channels; ++chan )
        {
            ++channelNo;

            // Check whether or not the channel is already a bad channel
            if( WBadChannelManager::instance()->isChannelBad(modality->getModalityType(), channelNo) )
            {
                continue; // skip this channel for processing
            }

            threshold = getThreshold( modality->getModalityType(), channelNo );

            WLEMData::ScalarT max = modality->getData().row( chan ).maxCoeff(); // maximum value
            WLEMData::ScalarT min = modality->getData().row( chan ).minCoeff(); // minimum value
            WLEMData::ScalarT diff = max - min; // difference between maximum and minimum

            if( diff > threshold )
            {
                // add a bad channel to the manager
                WBadChannelManager::instance()->addChannel( modality->getModalityType(), channelNo );

                m_BadChannelUpdated = true;

                ++rejections;
            }
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