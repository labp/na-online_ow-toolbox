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

#include <algorithm>
#include <list>
#include <map>
#include <string>
#include <vector>

#include <fiff/fiff_ch_info.h>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>

#include "core/container/WLArrayList.h"
#include "core/data/enum/WLEModality.h"
#include "core/dataFormat/fiff/WLFiffChType.h"
#include "modules/ftRtClient/reader/WReaderNeuromagHeader.h"
#include "WFTChunkNeuromagHdr.h"

const std::string WFTChunkNeuromagHdr::CLASS = "WFTChunkNeuromagHdr";

WFTChunkNeuromagHdr::WFTChunkNeuromagHdr( const char* data, const size_t size ) :
                WFTAChunk( WLEFTChunkType::FT_CHUNK_NEUROMAG_HEADER, size )
{
    processData( data, size );
}

boost::shared_ptr< const FIFFLIB::FiffInfo > WFTChunkNeuromagHdr::getData() const
{
    return m_measInfo;
}

WLArrayList< std::string >::SPtr WFTChunkNeuromagHdr::getChannelNames( WLEModality::Enum modality ) const
{
    if( m_measInfo == 0 )
    {
        return WLArrayList< std::string >::SPtr();
    }

    WLArrayList< std::string >::SPtr names( new WLArrayList< std::string > );

    for( int i = 0; i < m_measInfo->chs.size(); ++i )
    {
        if( modality == WLEModality::fromFiffType( m_measInfo->chs.at( i ).kind ) )
        {
            names->push_back( m_measInfo->chs.at( i ).ch_name.toStdString() );
        }
    }

    return names;
}

WFTChunkNeuromagHdr::ModalityPicks_SPtr WFTChunkNeuromagHdr::getModalityPicks() const
{
    return m_modalityPicks;
}

boost::shared_ptr< WLEMDRaw::ChanPicksT > WFTChunkNeuromagHdr::getStimulusPicks() const
{
    return m_stimulusPicks;
}

WLArrayList< WPosition >::SPtr WFTChunkNeuromagHdr::getChannelPositionsEEG() const
{
    if( hasChannelPositionsEEG() )
    {
        return m_chPosEEG;
    }

    return WLArrayList< WPosition >::instance();
}

WLArrayList< WPosition >::SPtr WFTChunkNeuromagHdr::getChannelPositionsMEG() const
{
    if( hasChannelPositionsMEG() )
    {
        return m_chPosMEG;
    }

    return WLArrayList< WPosition >::instance();
}

WLArrayList< WVector3f >::SPtr WFTChunkNeuromagHdr::getChannelExMEG() const
{
    if( !m_chExMEG || m_chExMEG->empty() )
    {
        return WLArrayList< WVector3f >::instance();
    }
    return m_chExMEG;
}

WLArrayList< WVector3f >::SPtr WFTChunkNeuromagHdr::getChannelEyMEG() const
{
    if( !m_chEyMEG || m_chEyMEG->empty() )
    {
        return WLArrayList< WVector3f >::instance();
    }
    return m_chEyMEG;
}

WLArrayList< WVector3f >::SPtr WFTChunkNeuromagHdr::getChannelEzMEG() const
{
    if( !m_chEzMEG || m_chEzMEG->empty() )
    {
        return WLArrayList< WVector3f >::instance();
    }
    return m_chEzMEG;
}

boost::shared_ptr< std::vector< float > > WFTChunkNeuromagHdr::getScaleFactors() const
{
    return m_scaleFactors;
}

bool WFTChunkNeuromagHdr::hasChannelPositionsEEG() const
{
    if( !m_chPosEEG )
    {
        return false;
    }

    return m_chPosEEG->size() > 0 && m_chPosEEG->empty() == false;
}

bool WFTChunkNeuromagHdr::hasChannelPositionsMEG() const
{
    if( !m_chPosMEG )
    {
        return false;
    }

    return m_chPosMEG->size() > 0 && m_chPosMEG->empty() == false;
}

WLSmartStorage::ConstSPtr WFTChunkNeuromagHdr::serialize() const
{
    WLSmartStorage::SPtr store( new WLSmartStorage );

    return store;
}

bool WFTChunkNeuromagHdr::process( const char* data, size_t size )
{
    wlog::debug( CLASS ) << "process() called.";

    m_measInfo.reset( new FIFFLIB::FiffInfo );
    m_modalityPicks.reset( new ModalityPicksT );
    m_stimulusPicks.reset( new WLEMDRaw::ChanPicksT );
    m_chPosEEG.reset( new WLArrayList< WPosition > );
    m_chPosMEG.reset( new WLArrayList< WPosition > );
    m_chExMEG.reset( new WLArrayList< WVector3f > );
    m_chEyMEG.reset( new WLArrayList< WVector3f > );
    m_chEzMEG.reset( new WLArrayList< WVector3f > );
    m_scaleFactors.reset( new std::vector< float > );

    WReaderNeuromagHeader::SPtr reader( new WReaderNeuromagHeader( data, size ) );

    if( !reader->read( m_measInfo.get() ) )
    {
        wlog::error( CLASS ) << "Neuromag header file could not read.";
        return false;
    }

    m_scaleFactors->clear();
    m_scaleFactors->reserve( m_measInfo->nchan );

    //
    //  Process channel information.
    //
    for( int i = 0; i < m_measInfo->chs.size(); ++i )
    {
        FIFFLIB::FiffChInfo info = m_measInfo->chs.at( i );
        WLEModality::Enum modalityType = WLEModality::fromFiffType( info.kind );

        //
        // Create pick vectors for all channel types.
        //
        WLEMDRaw::ChanPicksT *vector;
        if( info.kind == WLFiffLib::ChType::STIM ) // use stimulus channels
        {
            vector = m_stimulusPicks.get();
        }
        else // EEG / MEG data channels
        {
            if( m_modalityPicks->count( modalityType ) == 0 )
            {
                m_modalityPicks->insert(
                                std::map< WLEModality::Enum, WLEMDRaw::ChanPicksT >::value_type( modalityType,
                                                WLEMDRaw::ChanPicksT() ) );
            }

            vector = &m_modalityPicks->at( modalityType );
        }

        vector->conservativeResize( vector->cols() + 1 );
        ( *vector )[vector->cols() - 1] = i;

        //
        // Read the head positions for EEG and MEG.
        //
        if( modalityType == WLEModality::EEG )
        {
            const Eigen::Matrix< double, 3, 2, Eigen::DontAlign >& chPos = info.eeg_loc;
            const WPosition pos( chPos( 0, 0 ), chPos( 1, 0 ), chPos( 2, 0 ) );
            m_chPosEEG->push_back( pos );
        }
        if( modalityType == WLEModality::MEG )
        {
            const Eigen::Matrix< double, 12, 1, Eigen::DontAlign >& chPos = info.loc;
            const WPosition pos( chPos( 0, 0 ), chPos( 1, 0 ), chPos( 2, 0 ) );
            m_chPosMEG->push_back( pos );

            const WVector3f ex( chPos( 3, 0 ), chPos( 4, 0 ), chPos( 5, 0 ) );
            m_chExMEG->push_back( ex );
            const WVector3f ey( chPos( 6, 0 ), chPos( 7, 0 ), chPos( 8, 0 ) );
            m_chEyMEG->push_back( ey );
            const WVector3f ez( chPos( 9, 0 ), chPos( 10, 0 ), chPos( 11, 0 ) );
            m_chEzMEG->push_back( ez );
        }

        //
        // Scale factors.
        //
        if( info.range != 0 && info.cal != 0 )
        {
            m_scaleFactors->push_back( info.range * info.cal );
        }
    }

    //
    // Validate the read measurement information
    //
    WAssertDebug( m_chPosEEG->size() == m_modalityPicks->at( WLEModality::EEG ).cols(),
                    "Wrong number of EEG sensor positions or channel picks" );
    WAssertDebug( m_modalityPicks->at( WLEModality::MEG ).cols() % 3 == 0, "Wrong number of MEG channel picks" );
    WAssertDebug( m_chPosMEG->size() == m_modalityPicks->at( WLEModality::MEG ).cols(),
                    "Wrong number of MEG sensor positions or channel picks" );

    return true;
}
