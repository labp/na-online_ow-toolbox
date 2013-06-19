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

#include <algorithm> // transform
#include <cstddef>
#include <functional> // plus, divide, bind
#include <string>

#include <core/common/WLogger.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/util/WLTimeProfiler.h"

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
    WLEMMeasurement::SPtr emmOut( new WLEMMeasurement( *m_emmSum ) );
    // TODO(pieloth): new profiler
    WLTimeProfiler::SPtr profiler( new WLTimeProfiler( CLASS, "lifetime" ) );
    profiler->start();
    emmOut->setTimeProfiler( profiler );

    WLEMData::SPtr emdSum;
    WLEMData::SPtr emdOut;
    size_t channels;
    size_t samples;
    for( size_t mod = 0; mod < m_emmSum->getModalityCount(); ++mod )
    {
        emdSum = m_emmSum->getModality( mod );
        emdOut = emdSum->clone();
        channels = emdSum->getData().size();
        emdOut->getData().resize( channels );

        samples = emdSum->getData().front().size();
        for( size_t chan = 0; chan < channels; ++chan )
        {
            emdOut->getData()[chan].resize( samples );
            std::transform( emdSum->getData()[chan].begin(), emdSum->getData()[chan].end(), emdOut->getData()[chan].begin(),
                            std::bind2nd( std::divides< double >(), m_count ) );
        }
        emmOut->addModality( emdOut );
    }

    return emmOut;
}

void WEpochAveragingTotal::addEmmSum( const WLEMMeasurement::ConstSPtr emm )
{
    WLTimeProfiler tp(CLASS, "addEmmSum");
    ++m_count;
    WLEMData::ConstSPtr emdIn;
    WLEMData::SPtr emdSum;
    size_t channels;

    for( size_t mod = 0; mod < emm->getModalityCount(); ++mod )
    {
        emdIn = emm->getModality( mod );
        emdSum = m_emmSum->getModality( mod );
        channels = emdIn->getData().size();
        for( size_t chan = 0; chan < channels; ++chan )
        {
            std::transform( emdIn->getData()[chan].begin(), emdIn->getData()[chan].end(), emdSum->getData()[chan].begin(),
                            emdSum->getData()[chan].begin(), std::plus< double >() );
        }
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
    size_t channels;
    size_t samples;
    for( size_t mod = 0; mod < emm->getModalityCount(); ++mod )
    {
        emd = emm->getModality( mod );
        emdSum = emd->clone();
        channels = emd->getData().size();
        emdSum->getData().resize( channels );
        samples = emd->getData().front().size();
        for( size_t chan = 0; chan < channels; ++chan )
        {
            emdSum->getData()[chan].resize( samples, 0 );
        }
        m_emmSum->addModality( emdSum );
    }
}

void WEpochAveragingTotal::reset()
{
    WEpochAveraging::reset();
    m_emmSum.reset();
}
