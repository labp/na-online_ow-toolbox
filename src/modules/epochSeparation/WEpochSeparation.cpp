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

#include <algorithm> // std::min
#include <cmath> // floor
#include <cstddef> // size_t, ptrdiff
#include <list>
#include <set>
#include <string>

#include <core/common/WAssert.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WEpochSeparation.h"

const std::string WEpochSeparation::CLASS = "WEpochSeparation";

WEpochSeparation::WEpochSeparation()
{
    reset();
}

WEpochSeparation::WEpochSeparation( size_t channel, std::set< WLEMMeasurement::EventT > triggerMask, size_t preSamples,
                size_t postSamples ) :
                m_channel( channel ), m_triggerMask( triggerMask ), m_preSamples( preSamples ), m_postSamples( postSamples ), m_blockSize(
                                0 )
{
}

WEpochSeparation::~WEpochSeparation()
{
}

size_t WEpochSeparation::getChannel() const
{
    return m_channel;
}

void WEpochSeparation::setChannel( size_t channel )
{
    m_channel = channel;
}

std::set< WLEMMeasurement::EventT > WEpochSeparation::getTriggerMask() const
{
    return m_triggerMask;
}

void WEpochSeparation::setTriggerMask( std::set< WLEMMeasurement::EventT > triggerMask )
{
    m_triggerMask = triggerMask;
}

size_t WEpochSeparation::getPreSamples() const
{
    return m_preSamples;
}

void WEpochSeparation::setPreSamples( size_t preSamples )
{
    m_preSamples = preSamples;
}

size_t WEpochSeparation::getPostSamples() const
{
    return m_postSamples;
}

void WEpochSeparation::setPostSamples( size_t postSamples )
{
    m_postSamples = postSamples;
}

void WEpochSeparation::reset()
{
    setChannel( 0 );
    setPreSamples( 1 );
    setPostSamples( 1 );
    m_triggerMask.clear();
    m_blockSize = 0;
    m_buffer.reset();
    m_leftEpochs.clear();
    m_epochs.clear();
}

bool WEpochSeparation::hasEpochs() const
{
    return !m_epochs.empty();
}

size_t WEpochSeparation::epochSize() const
{
    return m_epochs.size();
}

WLEMMeasurement::SPtr WEpochSeparation::getNextEpoch()
{
    WLEMMeasurement::SPtr emm = m_epochs.front();
    m_epochs.pop_front();
    return emm;
}

size_t WEpochSeparation::extract( const WLEMMeasurement::SPtr emmIn )
{
    WLTimeProfiler tp( CLASS, "extract" );
    size_t count = 0;
    if( emmIn->getModalityCount() < 1 || emmIn->getEventChannelCount() < m_channel + 1 )
    {
        wlog::warn( CLASS ) << "No modality or event channel.";
        return count;
    }
    setupBuffer( emmIn->getModality( 0 ) );

    m_buffer->addData( emmIn );

    // Find trigger matches //
    std::list< size_t > indices;
    WLEMMeasurement::EChannelT& events = emmIn->getEventChannel( m_channel );

    WLEMMeasurement::EventT prevEvent = 0;
    for( size_t i = 0; i < events.size(); ++i )
    {
        for( std::set< WLEMMeasurement::EventT >::const_iterator mask = m_triggerMask.begin(); mask != m_triggerMask.end();
                        ++mask )
        {
            if( ( events[i] == *mask ) && prevEvent != *mask )
            {
                wlog::debug( CLASS ) << "Found event: " << i;
                indices.push_back( i );
            }
        }
        prevEvent = events[i];
    }

    // Process preSamples of found matches //
    for( std::list< size_t >::iterator it = indices.begin(); it != indices.end(); ++it )
    {
        try
        {
            LeftEpoch::SPtr leftEpoch = processPreSamples( *it );
            m_leftEpochs.push_back( leftEpoch );
        }
        catch( const WException& e )
        {
            wlog::info( CLASS ) << "Skipping found epoch! Not enough presamples.";
        }
    }

    // Process left epochs //
    std::list< LeftEpoch::SPtr >::iterator it = m_leftEpochs.begin();
    // while instead of for, due to avoid skipped items after an erase!
    while( it != m_leftEpochs.end() )
    {
        if( processPostSamples( *it, emmIn ) )
        {
            m_epochs.push_back( ( *it )->m_emm );
            it = m_leftEpochs.erase( it );
            ++count;
        }
        else
        {
            ++it;
        }
    }
    m_buffer->popData();

    return count;
}

inline size_t WEpochSeparation::nmod( ptrdiff_t a, size_t n )
{
    WAssertDebug( n > 0, "nmod(a, n): n must be greater than 0!" );
    if( a > 0 )
    {
        return a % n;
    }
    else
    {
        while( a < 0 )
        {
            a += n;
        }
        return a;
    }
}

void WEpochSeparation::setupBuffer( WLEMData::ConstSPtr emd )
{
    if( !m_buffer || m_blockSize == 0 )
    {
        wlog::debug( CLASS ) << "Creating new buffer ...";
        // have to save preSamples + 1 (current)
        // e.g. preSample = 500; blockSize = 100; elements = (500 + 100) / 100 = 6
        // ... 5x EMM for preSamples and 1x EMM for current sample in current EMM
        m_blockSize = emd->getSamplesPerChan();
        size_t elements = ceil( ( float )( m_preSamples + m_blockSize ) / m_blockSize );
        m_buffer.reset( new LaBP::WLRingBuffer< WLEMMeasurement >( elements ) );
        wlog::debug( CLASS ) << "BlockSize: " << m_blockSize;
        wlog::debug( CLASS ) << "Samples: " << m_preSamples + 1;
        wlog::debug( CLASS ) << "Space for EMM: " << elements;
        wlog::debug( CLASS ) << "Creating new buffer finished!";
    }
}

WEpochSeparation::LeftEpoch::SPtr WEpochSeparation::processPreSamples( size_t eIndex ) throw( WException )
{
    wlog::debug( CLASS ) << "processPreSamples() called!";
    WLTimeProfiler tp( CLASS, "processPreSamples" );
    LeftEpoch::SPtr leftEpoch( new LeftEpoch() );

    leftEpoch->m_emm = m_buffer->getData()->clone();
    leftEpoch->m_leftSamples = m_preSamples + 1 + m_postSamples;
    const WLEMMeasurement::SPtr emmEpoch = leftEpoch->m_emm;

    // Prepare modalities //
    const WLEMMeasurement::ConstSPtr emm = m_buffer->getData();
    WLEMData::ConstSPtr emd;
    WLEMData::SPtr emdEpoch;
    const size_t modalities = emm->getModalityCount();
    for( size_t mod = 0; mod < modalities; ++mod )
    {
        emd = emm->getModality( mod );
        emdEpoch = emd->clone();
        emdEpoch->getData().resize( emd->getNrChans(), m_preSamples + 1 + m_postSamples );
        emmEpoch->addModality( emdEpoch );
    }

    // Prepare event channels //
    const size_t eChannels = emm->getEventChannelCount();
    emmEpoch->getEventChannels()->resize( 1 );
    emmEpoch->getEventChannel( 0 ).reserve( m_preSamples + 1 + m_postSamples );
    WAssertDebug( emmEpoch->getEventChannel( 0 ).size() == 0, "emmEpoch->getEventChannel( chan ).size() == 0" );

    // Initialize indices //
    // pStart: Index for first samples in first packet
    size_t pStart = nmod( eIndex - m_preSamples, m_blockSize );
    // pIndex: Index of the first packet in (circular) buffer
    ptrdiff_t pIndex = static_cast< ptrdiff_t >( floor( ( double )( ( int )eIndex - ( int )m_preSamples ) / m_blockSize ) );
    wlog::debug( CLASS ) << "pIndex: " << pIndex << " - pStart: " << pStart;

    size_t samplesCopied = 0;
    size_t offset;
    while( samplesCopied < ( m_preSamples + 1 ) )
    {
        WAssertDebug( pStart < m_blockSize, "pStart < m_blockSize" );
        offset = std::min( m_blockSize - pStart, ( m_preSamples + 1 ) - samplesCopied );

        wlog::debug( CLASS ) << "m_buffer->size(): " << m_buffer->size();
        // Copy modalities //
        const WLEMMeasurement::ConstSPtr emm = m_buffer->getData( pIndex );
        WAssertDebug( emm, "m_buffer->getData(pIndex)" );

        const size_t modalities = emm->getModalityCount();
        WAssertDebug( modalities == emmEpoch->getModalityCount(), "Different modality Count!" );
        for( size_t mod = 0; mod < modalities; ++mod )
        {
            emd = emm->getModality( mod );
            emdEpoch = emmEpoch->getModality( mod );
            WLEMData::DataT& data = emdEpoch->getData();

            WAssertDebug( emd->getNrChans() == emdEpoch->getNrChans(), "Different channel count!" );
            const WLEMData::DataT::Index channels = data.rows();
            const WLEMData::DataT& dataIn = emd->getData();
            data.block( 0, samplesCopied, channels, offset ) = dataIn.block( 0, pStart, channels, offset );
        }

        // Copy event channels //
        boost::shared_ptr< WLEMMeasurement::EDataT > events = emmEpoch->getEventChannels();
        WAssertDebug( emmEpoch->getEventChannelCount() <= emm->getEventChannelCount(), "Different event channel count!" );
        events->at( 0 ).insert( events->at( 0 ).end(), emm->getEventChannel( m_channel ).begin() + pStart,
                        emm->getEventChannel( m_channel ).begin() + pStart + offset );

        pStart = 0;
        samplesCopied += offset;
        ++pIndex;
        wlog::debug( CLASS ) << "samples copied: " << offset;
    }

    leftEpoch->m_leftSamples = m_postSamples;
    leftEpoch->m_startIndex = eIndex + 1;

    wlog::debug( CLASS ) << "samplesCopied: " << samplesCopied;
    wlog::debug( CLASS ) << "processPreSamples() finished!";

    return leftEpoch;
}

bool WEpochSeparation::processPostSamples( LeftEpoch::SPtr leftEpoch, WLEMMeasurement::ConstSPtr emm )
{
    wlog::debug( CLASS ) << "processPostSamples() called!";
    WLTimeProfiler tp( CLASS, "processPostSamples" );

    WLEMMeasurement::SPtr emmEpoch = leftEpoch->m_emm;
    size_t samplesLeft = leftEpoch->m_leftSamples;
    size_t pStart = leftEpoch->m_startIndex;

    WLEMData::ConstSPtr emd;
    WLEMData::SPtr emdEpoch;

    size_t offset = std::min( samplesLeft, m_blockSize - pStart );
    WAssertDebug( pStart + offset <= m_blockSize, "pStart + offset <= blockSize" );

    const size_t modalities = emm->getModalityCount();
    WAssertDebug( emm->getModalityCount() == emmEpoch->getModalityCount(), "Different modality count!" );
    for( size_t mod = 0; mod < modalities; ++mod )
    {
        emd = emm->getModality( mod );
        emdEpoch = emmEpoch->getModality( mod );

        WLEMData::DataT& data = emdEpoch->getData();
        WAssertDebug( emd->getNrChans() == emdEpoch->getNrChans(), "Different channel count!" );
        const WLEMData::DataT::Index channels = data.rows();
        const WLEMData::DataT& dataIn = emd->getData();
        data.block( 0, data.cols() - samplesLeft, channels, offset ) = dataIn.block( 0, pStart, channels, offset );
    }

    boost::shared_ptr< WLEMMeasurement::EDataT > events = emm->getEventChannels();
    boost::shared_ptr< WLEMMeasurement::EDataT > eventsEpoch = emmEpoch->getEventChannels();
    WAssertDebug( emmEpoch->getEventChannelCount() <= emm->getEventChannelCount(), "Different event channel count!" );
    eventsEpoch->at( 0 ).insert( eventsEpoch->at( 0 ).end(), events->at( m_channel ).begin() + pStart,
                    events->at( m_channel ).begin() + pStart + offset );

    samplesLeft -= offset;
    leftEpoch->m_leftSamples = samplesLeft;
    leftEpoch->m_startIndex = 0;
    if( samplesLeft == 0 )
    {
        emmEpoch->setProfiler( emm->getProfiler()->clone() );
    }

    wlog::debug( CLASS ) << "samples copied: " << offset;
    wlog::debug( CLASS ) << "samplesLeft: " << samplesLeft;
    wlog::debug( CLASS ) << "processPostSamples() finished!";
    return samplesLeft == 0;
}
