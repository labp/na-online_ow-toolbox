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
    WLTimeProfiler tp( CLASS, "baseline" );
    WLEMData::ConstSPtr emd;

    WLEMMeasurement::SPtr emmOut = emm->clone();
    WLEMData::SPtr emdOut;

    for( size_t mod = 0; mod < emm->getModalityCount(); ++mod )
    {
        emd = emm->getModality( mod );
        WLEMData::DataT& data = emd->getData();
        emdOut = emd->clone();
        WLEMData::DataT& dataOut = emdOut->getData();
        dataOut = data;

        const WLChanNrT channels = emd->getNrChans();
        const WLSampleNrT tbase = std::min( m_tbase, emd->getSamplesPerChan() );

        WLEMData::SampleT means( channels );
        for( WLChanIdxT chan = 0; chan < channels; ++chan )
        {
            WLEMData::ScalarT mean = 0;
            for( WLSampleIdxT smp = 0; smp < tbase; ++smp )
            {
                mean += data( chan, smp );
            }
            mean = tbase > 0 ? ( mean / tbase ) : 0;
            means( chan ) = mean;
        }

        dataOut.colwise() -= means;

        emmOut->addModality( emdOut );
    }

    boost::shared_ptr< WLEMMeasurement::EDataT > events = emm->getEventChannels();
    boost::shared_ptr< WLEMMeasurement::EDataT > eventsOut = emmOut->getEventChannels();
    eventsOut->assign( events->begin(), events->end() );

    return emmOut;
}
