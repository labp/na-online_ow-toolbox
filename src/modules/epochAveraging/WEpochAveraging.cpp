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

#include <algorithm> // min()
#include <cstddef>
#include <string>
#include <vector>

#include "core/common/WLogger.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WEpochAveraging.h"

const std::string WEpochAveraging::CLASS = "WEpochAveraging";

WEpochAveraging::WEpochAveraging( size_t tbase ) :
                m_tbase( tbase )
{
    reset();
}

WEpochAveraging::~WEpochAveraging()
{
}

size_t WEpochAveraging::getCount() const
{
    return m_count;
}

void WEpochAveraging::reset()
{
    wlog::debug( CLASS ) << "reset() called!";
    m_count = 0;
}

size_t WEpochAveraging::getTBase() const
{
    return m_tbase;
}

void WEpochAveraging::setTBase( size_t tbase, bool reset )
{
    if( reset )
    {
        this->reset();
    }
    m_tbase = tbase;
}

WLEMMeasurement::SPtr WEpochAveraging::baseline( WLEMMeasurement::ConstSPtr emm )
{
    WLTimeProfiler tp(CLASS, "baseline");
    WLEMData::ConstSPtr emd;

    WLEMMeasurement::SPtr emmOut = emm->clone();
    WLEMData::SPtr emdOut;

    for( size_t mod = 0; mod < emm->getModalityCount(); ++mod )
    {
        std::vector< double > means;

        emd = emm->getModality( mod );
        WLEMData::DataT& data = emd->getData();

        emdOut = emd->clone();
        WLEMData::DataT& dataOut = emdOut->getData();
        dataOut.assign( data.begin(), data.end() );

        const size_t channels = emd->getNrChans();
        const size_t tbase = std::min( m_tbase, emd->getSamplesPerChan() );
        for( size_t chan = 0; chan < channels; ++chan )
        {
            double mean = 0;
            for( size_t smp = 0; smp < tbase; ++smp )
            {
                mean += data[chan][smp];
            }
            mean = tbase > 0 ? ( mean / tbase ) : 0;
            means.push_back( mean );
        }

        const size_t smpPerChan = emdOut->getSamplesPerChan();
        for( size_t chan = 0; chan < channels; ++chan )
        {
            double mean = means[chan];
            for( size_t smp = 0; smp < smpPerChan; ++smp )
            {
                dataOut[chan][smp] -= mean;
            }
        }
        emmOut->addModality( emdOut );
    }

    boost::shared_ptr< WLEMMeasurement::EDataT > events = emm->getEventChannels();
    boost::shared_ptr< WLEMMeasurement::EDataT > eventsOut = emmOut->getEventChannels();
    eventsOut->assign( events->begin(), events->end() );

    return emmOut;
}
