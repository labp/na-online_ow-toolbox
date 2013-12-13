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

#include "core/common/WLogger.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDMEG.h"
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

bool WEpochRejectionTotal::doRejection( const WLEMMeasurement::ConstSPtr emm )
{
    size_t it;
    bool rejection = false;

    wlog::debug( CLASS ) << "starting total channel rejection";

    for( it = 0; it < emm->getModalityCount(); ++it )
    {
        WLEMData::ConstSPtr modality = emm->getModality( it );
        WLEMDMEG::ConstSPtr meg;

        switch( modality->getModalityType() )
        {
            case LaBP::WEModalityType::EEG:
                rejection = rejection || calcRejection( modality->getDataBadChannels(), m_eegThreshold );
                break;
            case LaBP::WEModalityType::MEG:

                meg = boost::static_pointer_cast< const WLEMDMEG >( modality );

                rejection = rejection || calcRejection( meg->getDataBadChannels( LaBP::WEGeneralCoilType::MAGNETOMETER ), m_megMagThreshold );
                rejection = rejection || calcRejection( meg->getDataBadChannels( LaBP::WEGeneralCoilType::GRADIOMETER ), m_megGradThreshold );

                break;
            case LaBP::WEModalityType::EOG:
                rejection = rejection || calcRejection( modality->getDataBadChannels(), m_eogThreshold );
                break;
            default:
                break;
        }
    }

    wlog::debug( CLASS ) << "finished total channel rejection: " << rejection;

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
