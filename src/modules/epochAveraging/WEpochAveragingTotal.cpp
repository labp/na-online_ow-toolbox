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

#include <cstddef>
#include <string>

#include <core/common/WLogger.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WEpochAveraging.h"
#include "WEpochAveragingTotal.h"

const std::string WEpochAveragingTotal::CLASS = "WEpochAveragingTotal";

WEpochAveragingTotal::WEpochAveragingTotal( size_t tbase ) :
                WEpochAveraging( tbase )
{
}

WEpochAveragingTotal::~WEpochAveragingTotal()
{
}

WLEMMeasurement::SPtr WEpochAveragingTotal::getAverage( WLEMMeasurement::ConstSPtr emm )
{
    WLTimeProfiler tp( CLASS, "getAverage" );

    emm = WEpochAveraging::baseline( emm );

    checkEmmSum( emm );
    addEmmSum( emm );

    // Create output emm and divide data by count
    WLEMMeasurement::SPtr emmOut( new WLEMMeasurement( *emm ) );

    WLEMData::ConstSPtr emdSum;
    WLEMData::SPtr emdOut;
    for( size_t mod = 0; mod < m_emmSum->getModalityCount(); ++mod )
    {
        emdSum = m_emmSum->getModality( mod );
        emdOut = emdSum->clone();

        const WLEMData::DataT& dataIn = emdSum->getData();
        WLEMData::DataT& dataOut = emdOut->getData();
        dataOut = ( 1.0 / m_count ) * dataIn;

        emmOut->addModality( emdOut );
    }

    return emmOut;
}

void WEpochAveragingTotal::addEmmSum( const WLEMMeasurement::ConstSPtr emm )
{
    WLTimeProfiler tp( CLASS, "addEmmSum" );
    ++m_count;
    WLEMData::ConstSPtr emdIn;
    WLEMData::SPtr emdSum;

    for( size_t mod = 0; mod < emm->getModalityCount(); ++mod )
    {
        emdIn = emm->getModality( mod );
        emdSum = m_emmSum->getModality( mod );
        const WLEMData::DataT& dataIn = emdIn->getData();
        WLEMData::DataT& dataSum = emdSum->getData();
        dataSum += dataIn;
    }
}

void WEpochAveragingTotal::checkEmmSum( const WLEMMeasurement::ConstSPtr emm )
{
    if( m_emmSum )
    {
        return;
    }

    wlog::debug( CLASS ) << "Creating new emmSum";
    m_emmSum.reset( new WLEMMeasurement( *emm ) );

    WLEMData::ConstSPtr emd;
    WLEMData::SPtr emdSum;
    for( size_t mod = 0; mod < emm->getModalityCount(); ++mod )
    {
        emd = emm->getModality( mod );
        emdSum = emd->clone();
        const size_t channels = emd->getNrChans();
        const size_t samples = emd->getSamplesPerChan();
        emdSum->getData().setZero( channels, samples );

        m_emmSum->addModality( emdSum );
    }
}

void WEpochAveragingTotal::reset()
{
    WEpochAveraging::reset();
    m_emmSum.reset();
}
