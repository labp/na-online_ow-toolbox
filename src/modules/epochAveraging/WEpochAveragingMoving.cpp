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
#include <functional> // divides, plus, bind
#include <string>

#include "core/util/WLTimeProfiler.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"

#include "WEpochAveraging.h"
#include "WEpochAveragingMoving.h"

const std::string WEpochAveragingMoving::CLASS = "WEpochAveragingMoving";

WEpochAveragingMoving::WEpochAveragingMoving( size_t tbase, size_t size ) :
                WEpochAveraging( tbase )
{
    setSize( size );
}

WEpochAveragingMoving::~WEpochAveragingMoving()
{
}

size_t WEpochAveragingMoving::getSize() const
{
    return m_size;
}

size_t WEpochAveragingMoving::getCount() const
{
    return std::min( m_size, m_count );
}

WLEMMeasurement::SPtr WEpochAveragingMoving::getAverage( WLEMMeasurement::ConstSPtr emmIn )
{
    LaBP::WLTimeProfiler time( CLASS, "average" );
    time.start();

    emmIn = WEpochAveraging::baseline( emmIn );

    pushBuffer( emmIn );

    WLEMMeasurement::SPtr emmOut( new WLEMMeasurement( *emmIn ) );
    LaBP::WLTimeProfiler::SPtr profiler( new LaBP::WLTimeProfiler( CLASS, "lifetime" ) );
    profiler->start();
    emmOut->setTimeProfiler( profiler );

    WLEMData::ConstSPtr emdIn;
    WLEMData::SPtr emdOut;

    // Create output EMM
    size_t channels;
    size_t samples;
    for( size_t mod = 0; mod < emmIn->getModalityCount(); ++mod )
    {
        emdIn = emmIn->getModality( mod );
        emdOut = emdIn->clone();
        channels = emdIn->getData().size();
        emdOut->getData().resize( channels );
        samples = emdIn->getData().front().size();
        for( size_t chan = 0; chan < channels; ++chan )
        {
            emdOut->getData()[chan].resize( samples, 0 );
        }
        emmOut->addModality( emdOut );
    }

    // Get Sum
    for( size_t i = 0; i < std::min( m_size, m_count ); ++i ) // min() if buffer is not filled
    {
        emmIn = m_buffer[i];
        WAssertDebug( emmIn->getModalityCount() == emmOut->getModalityCount(), "Different modality count!" );
        for( size_t mod = 0; mod < emmIn->getModalityCount(); ++mod )
        {
            emdIn = emmIn->getModality( mod );
            emdOut = emmOut->getModality( mod );
            channels = emdIn->getData().size();
            WAssertDebug( channels == emdOut->getData().size(), "channels == emdOut->getData().size()" );
            for( size_t chan = 0; chan < channels; ++chan )
            {
                std::transform( emdIn->getData()[chan].begin(), emdIn->getData()[chan].end(), emdOut->getData()[chan].begin(),
                                emdOut->getData()[chan].begin(), std::plus< double >() );
            }
        }
    }

    // divide
    for( size_t mod = 0; mod < emmOut->getModalityCount(); ++mod )
    {
        emdOut = emmOut->getModality( mod );
        channels = emdOut->getData().size();
        for( size_t chan = 0; chan < channels; ++chan )
        {
            std::transform( emdOut->getData()[chan].begin(), emdOut->getData()[chan].end(), emdOut->getData()[chan].begin(),
                            std::bind2nd( std::divides< double >(), std::min( m_size, m_count ) ) );
        }
    }

    time.stopAndLog();

    return emmOut;
}

void WEpochAveragingMoving::setSize( size_t size )
{
    if( size != getSize() || m_buffer.size() != getSize() )
    {
        this->m_size = size;
        m_count = 0;
        m_ptr = 0;
        m_buffer.clear();
        m_buffer.resize( size );
    }
}

void WEpochAveragingMoving::pushBuffer( const WLEMMeasurement::ConstSPtr emm )
{
    ++m_count;
    m_buffer[m_ptr++] = emm;
    m_ptr = m_ptr % m_buffer.size();
}

void WEpochAveragingMoving::reset()
{
    WEpochAveraging::reset();
    m_ptr = 0;
}
