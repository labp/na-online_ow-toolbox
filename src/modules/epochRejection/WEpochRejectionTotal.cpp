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

#include <boost/shared_ptr.hpp>
#include <boost/foreach.hpp>

#include "core/common/WLogger.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDMEG.h"
#include "core/data/enum/WLEModality.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WBadChannelManager.h"
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

void WEpochRejectionTotal::initRejection()
{
    m_rejCount = 0;
}

bool WEpochRejectionTotal::doRejection( const WLEMMeasurement::ConstSPtr emm )
{
    size_t it;
    bool rejection = false;

    wlog::debug( CLASS ) << "starting total channel rejection.";

    initRejection();

    for( it = 0; it < emm->getModalityCount(); ++it )
    {
        WLEMData::ConstSPtr modality = emm->getModality( it );
        WLEMDMEG::ConstSPtr meg;

        if( !validModality( modality->getModalityType() ) ) // if wrong modality, next
        {
            continue;
        }

        if( WLEModality::isMEG( modality->getModalityType() ) ) // MEG modality
        {
            meg = boost::static_pointer_cast< const WLEMDMEG >( modality );

            rejection |= calcRejection(
                            meg->getDataBadChannels( WLEMEGGeneralCoilType::MAGNETOMETER,
                                            WBadChannelManager::instance()->getChannelList( modality->getModalityType() ) ),
                            getThreshold( WLEModality::MEG_MAG ) );

            rejection |= calcRejection(
                            meg->getDataBadChannels( WLEMEGGeneralCoilType::GRADIOMETER,
                                            WBadChannelManager::instance()->getChannelList( modality->getModalityType() ) ),
                            getThreshold( WLEModality::MEG_GRAD ) );
        }
        else // all other modalities
        {
            rejection |= calcRejection(
                            modality->getDataBadChannels(
                                            WBadChannelManager::instance()->getChannelList( modality->getModalityType() ) ),
                            getThreshold( modality->getModalityType() ) );
        }

        if( rejection )
        {
            wlog::debug( CLASS ) << "modality rejected: " << WLEModality::name( modality->getModalityType() );
            ++m_rejCount;
            break;
        }
    }

    wlog::debug( CLASS ) << "finished total channel rejection (result [true/false]: " << rejection << ").";

    return rejection;
}

bool WEpochRejectionTotal::calcRejection( const WLEMData::DataT& data, double threshold )
{
    // calculate the maximal difference between maximum peek and minimum peek over all channels
    WLEMData::ScalarT diff = data.maxCoeff() - data.minCoeff();

    return diff > threshold;
}

bool WEpochRejectionTotal::calcRejection( const WLEMData::DataSPtr data, double threshold )
{
    // calculate the maximal difference between maximum peek and minimum peek over all channels
    WLEMData::ScalarT diff = data->maxCoeff() - data->minCoeff();

    return diff > threshold;
}
