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
    m_modalityPicks.reset( new ModalityPicksT );
    m_stimulusPicks.reset( new WLEMDRaw::ChanPicksT );
    m_chPosEEG.reset( new WLArrayList< WPosition > );
    m_chPosMEG.reset( new WLArrayList< WPosition > );
    m_chExMEG.reset( new WLArrayList< WVector3f > );
    m_chEyMEG.reset( new WLArrayList< WVector3f > );
    m_chEzMEG.reset( new WLArrayList< WVector3f > );
    m_scaleFactors.reset( new std::vector< float > );

    WReaderNeuromagHeader::SPtr reader( new WReaderNeuromagHeader( ( const char* )chunk->getData(), chunk->getDataSize() ) );

    if( !reader->read( m_measInfo.data() ) )
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
        if( info.range != 0.0 && info.cal != 0.0 )
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
        WLArrayList< std::string >::SPtr chNames = getChannelNames( mod );
        if( !chNames->empty() )
        {
            emd->setChanNames( chNames );
        }

        // Apply scaling
        if( m_applyScaling )
        {
            if( m_modalityPicks->count( mod ) > 0 )
            {
                const WLEMDRaw::ChanPicksT& picks = ( *m_modalityPicks )[mod];
                if( picks.size() == emd->getNrChans() )
                {
                    for( WLEMDRaw::ChanPicksT::Index row = 0; row < picks.size(); ++row )
                    {
                        emd->getData().row( row ) *= getScaleFactors()->at( picks[row] );
                    }
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
            continue;
        }
        // Set MEG information
        if( mod == WLEModality::MEG )
        {
            WLEMDMEG::SPtr meg = boost::dynamic_pointer_cast< WLEMDMEG >( emd );
            if( !m_chPosMEG->empty() )
            {
                meg->setChannelPositions3d( m_chPosMEG );
            }
            if( !m_chExMEG->empty() && !m_chEyMEG->empty() && !m_chEzMEG->empty() )
            {
                meg->setEx( m_chExMEG );
                meg->setEy( m_chEyMEG );
                meg->setEz( m_chEzMEG );
            }
            continue;
        }
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

FIFFLIB::FiffInfo::ConstSPtr WFTChunkReaderNeuromagHdr::getMeasInfo() const
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

void WFTChunkReaderNeuromagHdr::setApplyScaling( bool apply )
{
    m_applyScaling = apply;
}

bool WFTChunkReaderNeuromagHdr::extractEmdsByPicks( WLEMMeasurement::SPtr emm, WLEMDRaw::ConstSPtr raw )
{
    if( m_modalityPicks->empty() )
    {
        wlog::error( CLASS ) << __func__ << ": No modality picks available!";
        return false;
    }

    bool rc = false;
    WLEMData::SPtr emd;
    WLEMData::DataSPtr data;

    if( m_modalityPicks->count( WLEModality::EEG ) > 0 )
    {
        emd.reset( new WLEMDEEG() );
        rc |= extractEmdsByPicks( emm, emd, raw );
    }

    if( m_modalityPicks->count( WLEModality::MEG ) > 0 )
    {
        emd.reset( new WLEMDMEG() );
        rc |= extractEmdsByPicks( emm, emd, raw );
    }

    if( m_modalityPicks->count( WLEModality::EOG ) > 0 )
    {
        emd.reset( new WLEMDEOG() );
        rc |= extractEmdsByPicks( emm, emd, raw );
    }

    if( m_modalityPicks->count( WLEModality::ECG ) > 0 )
    {
        emd.reset( new WLEMDECG() );
        rc |= extractEmdsByPicks( emm, emd, raw );
    }

    return rc;
}

bool WFTChunkReaderNeuromagHdr::extractEmdsByPicks( WLEMMeasurement::SPtr emm, WLEMData::SPtr emd, WLEMDRaw::ConstSPtr raw )
{
    const WLEMDRaw::ChanPicksT& picks = ( *m_modalityPicks )[emd->getModalityType()];
    WLEMData::DataSPtr data;
    try
    {
        data = raw->getData( picks );
    }
    catch( const WOutOfBounds& e )
    {
        wlog::error( CLASS ) << __func__ << ": " << e.what();
        return false;
    }
    emd->setData( data );
    emd->setSampFreq( raw->getSampFreq() );
    emm->addModality( emd );
    return true;
}
