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
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDECG.h"
#include "core/data/emd/WLEMDEOG.h"
#include "core/data/emd/WLEMDMEG.h"
#include "core/data/enum/WLEModality.h"
#include "core/dataFormat/fiff/WLFiffChType.h"
#include "modules/ftRtClient/ftb/WFtbChunk.h"
#include "modules/ftRtClient/reader/WReaderNeuromagHeader.h"
#include "WFTChunkReaderNeuromagHdr.h"

const std::string WFTChunkReaderNeuromagHdr::CLASS = "WFTChunkReaderNeuromagHdr";

WFTChunkReaderNeuromagHdr::WFTChunkReaderNeuromagHdr()
{
    m_measInfo.reset( new FIFFLIB::FiffInfo );
    m_chPosEEG = WLPositions::instance();
    m_chPosMEG = WLPositions::instance();
    m_chExMEG.reset( new WLArrayList< WVector3f > );
    m_chEyMEG.reset( new WLArrayList< WVector3f > );
    m_chEzMEG.reset( new WLArrayList< WVector3f > );
    m_applyScaling = false;
}

WFTChunkReaderNeuromagHdr::~WFTChunkReaderNeuromagHdr()
{
}

wftb::chunk_type_t WFTChunkReaderNeuromagHdr::supportedChunkType() const
{
    return wftb::ChunkType::NEUROMAG_HEADER;
}

bool WFTChunkReaderNeuromagHdr::read( WFTChunk::ConstSPtr chunk )
{
    wlog::debug( CLASS ) << __func__ << "() called.";

    if( chunk->getChunkType() != supportedChunkType() )
    {
        wlog::error( CLASS ) << "Chunk type not supported: " << wftb::ChunkType::name( chunk->getChunkType() );
        return false;
    }

    m_measInfo.reset( new FIFFLIB::FiffInfo );
    m_modalityPicks.clear();
    m_stimulusPicks.resize( 0 );

    m_chPosEEG = WLPositions::instance();
    // TODO(pieloth): #393 set unit and exponent.
    m_chPosEEG->coordSystem( WLECoordSystem::HEAD );

    m_chPosMEG = WLPositions::instance();
    // TODO(pieloth): #393 set unit and exponent.
    m_chPosMEG->coordSystem( WLECoordSystem::DEVICE );

    m_chExMEG.reset( new WLArrayList< WVector3f > );
    m_chEyMEG.reset( new WLArrayList< WVector3f > );
    m_chEzMEG.reset( new WLArrayList< WVector3f > );

    WReaderNeuromagHeader::SPtr reader( new WReaderNeuromagHeader( ( const char* )chunk->getData(), chunk->getDataSize() ) );

    if( !reader->read( m_measInfo.data() ) )
    {
        wlog::error( CLASS ) << "Neuromag header file could not read.";
        return false;
    }

    m_scaleFactors.clear();
    m_scaleFactors.reserve( m_measInfo->nchan );

    //
    //  Process channel information.
    //
    WLPositions::IndexT idxEEG = 0;
    WLPositions::IndexT idxMEG = 0;
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
            vector = &m_stimulusPicks;
        }
        else // EEG / MEG data channels
        {
            if( m_modalityPicks.count( modalityType ) == 0 )
            {
                m_modalityPicks.insert( ModalityPicksT::value_type( modalityType, WLEMDRaw::ChanPicksT() ) );
            }
            vector = &m_modalityPicks.at( modalityType );

            if( m_modalityChNames.count( modalityType ) == 0 )
            {
                m_modalityChNames.insert( ModalityChNamesT::value_type( modalityType, WLArrayList< std::string >::instance() ) );
            }
            m_modalityChNames[modalityType]->push_back( m_measInfo->chs.at( i ).ch_name.toStdString() );
        }

        vector->conservativeResize( vector->cols() + 1 );
        ( *vector )[vector->cols() - 1] = i;

        //
        // Read the head positions for EEG and MEG.
        //
        if( modalityType == WLEModality::EEG )
        {
            const Eigen::Matrix< double, 3, 2, Eigen::DontAlign >& chPos = info.eeg_loc;
            if( m_chPosEEG->empty() )
            {
                m_chPosEEG->resize( m_modalityPicks.at( WLEModality::EEG ).cols() );
            }
            m_chPosEEG->data().col( idxEEG ).x() = chPos( 0, 0 );
            m_chPosEEG->data().col( idxEEG ).x() = chPos( 1, 0 );
            m_chPosEEG->data().col( idxEEG ).x() = chPos( 2, 0 );
            ++idxEEG;
        }
        if( modalityType == WLEModality::MEG )
        {
            const Eigen::Matrix< double, 12, 1, Eigen::DontAlign >& chPos = info.loc;
            if( m_chPosMEG->empty() )
            {
                m_chPosMEG->resize( m_modalityPicks.at( WLEModality::MEG ).cols() );
            }
            m_chPosMEG->data().col( idxMEG ).x() = chPos( 0, 0 );
            m_chPosMEG->data().col( idxMEG ).x() = chPos( 1, 0 );
            m_chPosMEG->data().col( idxMEG ).x() = chPos( 2, 0 );
            ++idxMEG;

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
        if( info.range != 0.0 && info.cal != 0.0 )
        {
            m_scaleFactors.push_back( info.range * info.cal );
        }
    }

    //
    // Validate the read measurement information
    //
    WAssertDebug( m_chPosEEG->size() == m_modalityPicks.at( WLEModality::EEG ).cols(),
                    "Wrong number of EEG sensor positions or channel picks" );
    WAssertDebug( m_modalityPicks.at( WLEModality::MEG ).cols() % 3 == 0, "Wrong number of MEG channel picks" );
    WAssertDebug( m_chPosMEG->size() == m_modalityPicks.at( WLEModality::MEG ).cols(),
                    "Wrong number of MEG sensor positions or channel picks" );

    return true;
}

bool WFTChunkReaderNeuromagHdr::apply( WLEMMeasurement::SPtr emm, WLEMDRaw::SPtr rawData )
{
    bool rc = false;

    if( emm->getModalityCount() == 0 )
    {
        if( !extractEmdsByPicks( emm, rawData ) )
        {
            wlog::error( CLASS ) << "Could not extract any modality!";
            return false;
        }
    }

    std::vector< WLEMData::SPtr > mods = emm->getModalityList();
    std::vector< WLEMData::SPtr >::iterator modIt;
    for( modIt = mods.begin(); modIt != mods.end(); ++modIt )
    {
        const WLEModality::Enum mod = ( *modIt )->getModalityType();
        const WLEMData::SPtr emd = *modIt;

        // Set Channels names
        if( m_modalityChNames.count( mod ) > 0 && !m_modalityChNames[mod]->empty() )
        {
            emd->setChanNames( m_modalityChNames[mod] );
            rc |= true;
        }

        // Apply scaling
        if( m_applyScaling )
        {
            if( m_modalityPicks.count( mod ) > 0 )
            {
                const WLEMDRaw::ChanPicksT& picks = m_modalityPicks[mod];
                if( picks.size() == emd->getNrChans() )
                {
                    for( WLEMDRaw::ChanPicksT::Index row = 0; row < picks.size(); ++row )
                    {
                        emd->getData().row( row ) *= m_scaleFactors.at( picks[row] );
                    }
                    rc |= true;
                }
                else
                {
                    wlog::error( CLASS ) << "Channel count and picks does not match!";
                }
            }
        }

        // Set EEG information
        if( mod == WLEModality::EEG && !m_chPosEEG->empty() )
        {
            WLEMDEEG::SPtr eeg = boost::dynamic_pointer_cast< WLEMDEEG >( emd );
            eeg->setChannelPositions3d( m_chPosEEG );
            rc |= true;
            continue;
        }
        // Set MEG information
        if( mod == WLEModality::MEG )
        {
            WLEMDMEG::SPtr meg = boost::dynamic_pointer_cast< WLEMDMEG >( emd );
            if( !m_chPosMEG->empty() )
            {
                meg->setChannelPositions3d( m_chPosMEG );
                rc |= true;
            }
            if( !m_chExMEG->empty() && !m_chEyMEG->empty() && !m_chEzMEG->empty() )
            {
                meg->setEx( m_chExMEG );
                meg->setEy( m_chEyMEG );
                meg->setEz( m_chEzMEG );
                rc |= true;
            }
            continue;
        }
    }

    //  Add event / stimulus channels to the EMM
    rc |= extractEventsByPicks( emm, rawData );

    return rc;
}

void WFTChunkReaderNeuromagHdr::setApplyScaling( bool apply )
{
    m_applyScaling = apply;
}
bool WFTChunkReaderNeuromagHdr::extractEventsByPicks( WLEMMeasurement::SPtr emm, WLEMDRaw::ConstSPtr raw )
{
    if( m_stimulusPicks.size() == 0 )
    {
        return false;
    }

    const WLEMDRaw::ChanPicksT& ePicks = m_stimulusPicks;
    const WLEMDRaw::DataT& rawData = raw->getData();

    const WLEMDRaw::DataT::Index rows = ePicks.size();
    const WLEMDRaw::DataT::Index cols = rawData.cols();

    boost::shared_ptr< WLEMMeasurement::EDataT > events( new WLEMMeasurement::EDataT );
    events->clear();
    events->reserve( rows );

    for( WLEMDRaw::DataT::Index row = 0; row < rows; ++row )
    {
        WLEMMeasurement::EChannelT eChannel;
        eChannel.reserve( cols );
        for( WLEMDRaw::DataT::Index col = 0; col < cols; ++col )
        {
            eChannel.push_back( ( WLEMMeasurement::EventT )rawData( ePicks[row], col ) );
        }
        events->push_back( eChannel );
    }

    emm->setEventChannels( events );
    return true;
}

bool WFTChunkReaderNeuromagHdr::extractEmdsByPicks( WLEMMeasurement::SPtr emm, WLEMDRaw::ConstSPtr raw )
{
    if( m_modalityPicks.empty() )
    {
        wlog::error( CLASS ) << __func__ << ": No modality picks available!";
        return false;
    }

    bool rc = false;

    std::list< WLEMData::SPtr > mods;
    if( m_modalityPicks.count( WLEModality::EEG ) > 0 )
    {
        mods.push_back( WLEMData::SPtr( new WLEMDEEG() ) );
    }
    if( m_modalityPicks.count( WLEModality::MEG ) > 0 )
    {
        mods.push_back( WLEMData::SPtr( new WLEMDMEG() ) );
    }
    if( m_modalityPicks.count( WLEModality::EOG ) > 0 )
    {
        mods.push_back( WLEMData::SPtr( new WLEMDEOG() ) );
    }
    if( m_modalityPicks.count( WLEModality::ECG ) > 0 )
    {
        mods.push_back( WLEMData::SPtr( new WLEMDECG() ) );
    }

    WLEMData::SPtr emd;
    WLEMData::DataSPtr data;
    for( std::list< WLEMData::SPtr >::iterator it = mods.begin(); it != mods.end(); ++it )
    {
        emd = *it;
        const WLEMDRaw::ChanPicksT& picks = m_modalityPicks[emd->getModalityType()];
        try
        {
            data = raw->getData( picks );
        }
        catch( const WOutOfBounds& e )
        {
            wlog::error( CLASS ) << __func__ << ": " << e.what();
            continue;
        }
        emd->setData( data );
        emd->setSampFreq( raw->getSampFreq() );
        emm->addModality( emd );
        rc |= true;
    }

    return rc;
}
