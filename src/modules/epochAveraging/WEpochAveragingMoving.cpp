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

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/util/profiler/WLTimeProfiler.h"

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
    WLTimeProfiler tp( CLASS, "getAverage" );

    emmIn = WEpochAveraging::baseline( emmIn );

    pushBuffer( emmIn );

    WLEMMeasurement::SPtr emmOut( new WLEMMeasurement( *emmIn ) );

    WLEMData::ConstSPtr emdIn;
    WLEMData::SPtr emdOut;

    // Create output EMM
    for( size_t mod = 0; mod < emmIn->getModalityCount(); ++mod )
    {
        emdIn = emmIn->getModality( mod );
        emdOut = emdIn->clone();
        const size_t channels = emdIn->getNrChans();
        const size_t samples = emdIn->getSamplesPerChan();
        emdOut->getData().setZero( channels, samples );
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
            WAssertDebug( emdIn->getNrChans() == emdOut->getNrChans(), "emdIn->getNrChans() == emdOut->getNrChans()" );

            const WLEMData::DataT& dataIn = emdIn->getData();
            WLEMData::DataT& dataOut = emdOut->getData();
            dataOut += dataIn;
        }
    }

    // divide
    WLEMData::SampleT divFactor;
    for( size_t mod = 0; mod < emmOut->getModalityCount(); ++mod )
    {
        emdOut = emmOut->getModality( mod );
        WLEMData::DataT& dataOut = emdOut->getData();
        divFactor = 1.0 / std::min( m_size, m_count );
        dataOut *= divFactor;
    }

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
