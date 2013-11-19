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
#include "WEpochRejectionTotal.h"

const std::string WEpochRejectionTotal::CLASS = "WEpochRejectionTotal";

WEpochRejectionTotal::WEpochRejectionTotal() :
                WEpochRejection::WEpochRejection()
{

}

WEpochRejectionTotal::~WEpochRejectionTotal()
{

}

/**
 * Method to process the rejection on the data.
 */
bool WEpochRejectionTotal::getRejection( const WLEMMeasurement::SPtr emm )
{
    WLTimeProfiler tp( "WEpochRejectionTotal", "getRejection" );
    wlog::debug( CLASS ) << "starting rejection for all channels";

    m_rejCount = 0;

    WLEMData::SPtr modality;

    for( size_t mod = 0; mod < emm->getModalityCount(); ++mod ) // for all modalities
    {
        // get modality
        modality = emm->getModality( mod );
        size_t rejections = 0;

        // if wrong modality, next
        if( !validModality( modality->getModalityType() ) )
        {
            wlog::debug( CLASS ) << "invalid modality";
            continue;
        }

        WLEMData::SampleT max = modality->getData().rowwise().maxCoeff(); // get the maximums for all channels
        WLEMData::SampleT min = modality->getData().rowwise().minCoeff(); // get the minimums for all channels
        WLEMData::SampleT diff = max - min; // calculate the difference for all channels

        // calculate the maximal difference between maximum peek and minimum peek over all channels
        WLEMData::ScalarT diffMax = max.maxCoeff() - min.minCoeff();

        // definition of the threshold to use by the modality
        switch( modality->getModalityType() )
        {
            case LaBP::WEModalityType::EEG:
                if( diffMax > m_eegLevel )
                {
                    ++rejections; // counts the rejected for each modality
                }
                break;
            case LaBP::WEModalityType::EOG:
                if( diffMax > m_eogLevel )
                {
                    ++rejections; // counts the rejected for each modality
                }
                break;
            default:
                ;
        }
    }

    return m_rejCount > 0;
}
