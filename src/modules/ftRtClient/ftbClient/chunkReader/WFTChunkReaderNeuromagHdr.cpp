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
    m_modalityPicks.reset( new ModalityPicksT );
    m_stimulusPicks.reset( new WLEMDRaw::ChanPicksT );
    m_chPosEEG.reset( new WLArrayList< WPosition > );
    m_chPosMEG.reset( new WLArrayList< WPosition > );
    m_chExMEG.reset( new WLArrayList< WVector3f > );
    m_chEyMEG.reset( new WLArrayList< WVector3f > );
    m_chEzMEG.reset( new WLArrayList< WVector3f > );
    m_scaleFactors.reset( new std::vector< float > );
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
    m_modalityPicks.reset( new ModalityPicksT );
    m_stimulusPicks.reset( new WLEMDRaw::ChanPicksT );
    m_chPosEEG.reset( new WLArrayList< WPosition > );
    m_chPosMEG.reset( new WLArrayList< WPosition > );
    m_chExMEG.reset( new WLArrayList< WVector3f > );
    m_chEyMEG.reset( new WLArrayList< WVector3f > );
    m_chEzMEG.reset( new WLArrayList< WVector3f > );
    m_scaleFactors.reset( new std::vector< float > );

    WReaderNeuromagHeader::SPtr reader( new WReaderNeuromagHeader( ( const char* )chunk->getData(), chunk->getDataSize() ) );

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

bool WFTChunkReaderNeuromagHdr::apply( WLEMMeasurement::SPtr emm, WLEMDRaw::SPtr rawData )
{
    bool rc = false;

    std::map< WLEModality::Enum, WLEMDRaw::ChanPicksT >::const_iterator it;
    for( it = getModalityPicks()->begin(); it != getModalityPicks()->end(); ++it )
    {
        // skip unknown modalities: in case of Neuromag - the "Misc" channel.
        if( it->first == WLEModality::UNKNOWN )
        {
            continue;
        }

        //
        // Create the data container.
        //
        WLEMData::SPtr emd;
        switch( it->first )
        {
            case WLEModality::EEG:
                emd.reset( new WLEMDEEG() );
                break;
            case WLEModality::MEG:
                emd.reset( new WLEMDMEG() );
                break;
            case WLEModality::EOG:
                emd.reset( new WLEMDEOG() );
                break;
            case WLEModality::ECG:
                emd.reset( new WLEMDECG() );
                break;
            case WLEModality::UNKNOWN:
                emd.reset( new WLEMDRaw() );
                break;
            default:
                continue;
        }

        emd->setData( rawData->getData( it->second, true ) );
        emd->setSampFreq( rawData->getSampFreq() );

        //
        // apply scaling
        //
        // TODO(pieloth): apply scaling
//            if( m_applyScaling )
//            {
//                for( Eigen::RowVectorXi::Index row = 0; row < it->second.size(); ++row )
//                {
//                    emd->getData().row( row ) *= neuromagHdr->getScaleFactors()->at( ( int )it->second[row] );
//                }
//            }

        //
        // Channel names.
        //
        WLArrayList< std::string >::SPtr channelNames = getChannelNames( it->first );
        if( channelNames )
        {
            if( channelNames->size() > 0 )
            {
                emd->setChanNames( channelNames );
            }
        }

        if( emd->getModalityType() == WLEModality::MEG ) // MEG
        {
            WLEMDMEG::SPtr meg = emd->getAs< WLEMDMEG >();
            if( hasChannelPositionsMEG() )
            {
                meg->setChannelPositions3d( getChannelPositionsMEG() );
            }
            meg->setEx( getChannelExMEG() );
            meg->setEy( getChannelEyMEG() );
            meg->setEz( getChannelEzMEG() );
        }

        emm->addModality( emd ); // add modality to measurement.
        rc |= true;
    }

    //
    //  Add event / stimulus channels to the EMM
    //
    if( getStimulusPicks()->cols() > 0 )
    {
        emm->setEventChannels( readEventChannels( ( Eigen::MatrixXf& )rawData->getData(), *getStimulusPicks() ) );
        rc |= true;
    }

    return rc;
}

boost::shared_ptr< const FIFFLIB::FiffInfo > WFTChunkReaderNeuromagHdr::getMeasInfo() const
{
    return m_measInfo;
}

WLArrayList< std::string >::SPtr WFTChunkReaderNeuromagHdr::getChannelNames( WLEModality::Enum modality ) const
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

WFTChunkReaderNeuromagHdr::ModalityPicks_SPtr WFTChunkReaderNeuromagHdr::getModalityPicks() const
{
    return m_modalityPicks;
}

boost::shared_ptr< WLEMDRaw::ChanPicksT > WFTChunkReaderNeuromagHdr::getStimulusPicks() const
{
    return m_stimulusPicks;
}

WLArrayList< WPosition >::SPtr WFTChunkReaderNeuromagHdr::getChannelPositionsEEG() const
{
    if( hasChannelPositionsEEG() )
    {
        return m_chPosEEG;
    }

    return WLArrayList< WPosition >::instance();
}

WLArrayList< WPosition >::SPtr WFTChunkReaderNeuromagHdr::getChannelPositionsMEG() const
{
    if( hasChannelPositionsMEG() )
    {
        return m_chPosMEG;
    }

    return WLArrayList< WPosition >::instance();
}

WLArrayList< WVector3f >::SPtr WFTChunkReaderNeuromagHdr::getChannelExMEG() const
{
    if( !m_chExMEG || m_chExMEG->empty() )
    {
        return WLArrayList< WVector3f >::instance();
    }
    return m_chExMEG;
}

WLArrayList< WVector3f >::SPtr WFTChunkReaderNeuromagHdr::getChannelEyMEG() const
{
    if( !m_chEyMEG || m_chEyMEG->empty() )
    {
        return WLArrayList< WVector3f >::instance();
    }
    return m_chEyMEG;
}

WLArrayList< WVector3f >::SPtr WFTChunkReaderNeuromagHdr::getChannelEzMEG() const
{
    if( !m_chEzMEG || m_chEzMEG->empty() )
    {
        return WLArrayList< WVector3f >::instance();
    }
    return m_chEzMEG;
}

boost::shared_ptr< std::vector< float > > WFTChunkReaderNeuromagHdr::getScaleFactors() const
{
    return m_scaleFactors;
}

bool WFTChunkReaderNeuromagHdr::hasChannelPositionsEEG() const
{
    if( !m_chPosEEG )
    {
        return false;
    }

    return m_chPosEEG->size() > 0 && m_chPosEEG->empty() == false;
}

bool WFTChunkReaderNeuromagHdr::hasChannelPositionsMEG() const
{
    if( !m_chPosMEG )
    {
        return false;
    }

    return m_chPosMEG->size() > 0 && m_chPosMEG->empty() == false;
}

boost::shared_ptr< WLEMMeasurement::EDataT > WFTChunkReaderNeuromagHdr::readEventChannels( const Eigen::MatrixXf& rawData,
                WLEMDRaw::ChanPicksT ePicks ) const
{
    //wlog::debug( CLASS ) << "readEventChannels() called.";

    boost::shared_ptr< WLEMMeasurement::EDataT > events( new WLEMMeasurement::EDataT );

    if( ePicks.size() == 0 )
    {
        wlog::error( CLASS ) << "No channels to pick.";
        return events;
    }

    const Eigen::RowVectorXi::Index rows = ePicks.size();
    const Eigen::MatrixXf::Index cols = rawData.cols();

    events->clear();
    events->reserve( rows );

    for( Eigen::RowVectorXi::Index row = 0; row < rows; ++row )
    {
        WLEMMeasurement::EChannelT eChannel;
        eChannel.reserve( cols );
        for( Eigen::RowVectorXi::Index col = 0; col < cols; ++col )
        {
            eChannel.push_back( ( WLEMMeasurement::EventT )rawData( ePicks[row], col ) );
        }
        events->push_back( eChannel );
    }

    return events;
}
