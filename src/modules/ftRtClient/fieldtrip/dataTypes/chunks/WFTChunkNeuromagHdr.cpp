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

#include <list>
#include <algorithm>

#include <boost/foreach.hpp>

#include <core/common/WLogger.h>
#include "core/container/WLArrayList.h"
#include "core/data/enum/WLEModality.h"
#include "core/dataFormat/fiff/WLFiffChType.h"

#include "modules/ftRtClient/reader/WReaderNeuromagHeader.h"
#include "WFTChunkNeuromagHdr.h"

const std::string WFTChunkNeuromagHdr::CLASS = "WFTChunkNeuromagHdr";

#ifdef _WIN32

const std::string WFTChunkNeuromagHdr::TMPDIRPATH = "C:/Windows/temp/";

#else

const std::string WFTChunkNeuromagHdr::TMPDIRPATH = "/tmp/";

#endif

const std::string WFTChunkNeuromagHdr::TMPFILENAME = TMPDIRPATH + "neuromag_header.fif";

WFTChunkNeuromagHdr::WFTChunkNeuromagHdr( const char* data, const size_t size ) :
                WFTAChunk( WLEFTChunkType::FT_CHUNK_NEUROMAG_HEADER, size )
{
    processData( data, size );
}

boost::shared_ptr< const FIFFLIB::FiffInfo > WFTChunkNeuromagHdr::getData() const
{
    return m_data;
}

WLArrayList< std::string >::SPtr WFTChunkNeuromagHdr::getChannelNames( WLEModality::Enum modality ) const
{
    if( m_data == 0 )
    {
        return WLArrayList< std::string >::SPtr();
    }

    WLArrayList< std::string >::SPtr names( new WLArrayList< std::string > );

    for( int i = 0; i < m_data->chs.size(); ++i )
    {

        if( modality == WLEModality::fromFiffType( m_data->chs.at( i ).kind ) )
        {
            names->push_back( m_data->chs.at( i ).ch_name.toStdString() );
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

WFTChunkNeuromagHdr::ChannelsPositionsSPtr WFTChunkNeuromagHdr::getChannelPositionsEEG() const
{
    return m_chPosEEG;
}

WFTChunkNeuromagHdr::ChannelsPositionsSPtr WFTChunkNeuromagHdr::getChannelPositionsMEG() const
{
    return m_chPosMEG;
}

boost::shared_ptr< std::vector< float > > WFTChunkNeuromagHdr::getScaleFactors() const
{
    return m_scaleFactors;
}

WLSmartStorage::ConstSPtr WFTChunkNeuromagHdr::serialize() const
{
    WLSmartStorage::SPtr store( new WLSmartStorage );

    // TODO(maschke): serialize measurement information into smart storage.

    return store;
}

bool WFTChunkNeuromagHdr::process( const char* data, size_t size )
{
    wlog::debug( CLASS ) << "process() called.";

    m_data.reset( new FIFFLIB::FiffInfo );
    m_modalityPicks.reset( new ModalityPicksT );
    m_stimulusPicks.reset( new WLEMDRaw::ChanPicksT );
    m_chPosEEG.reset( new ChannelsPositionsT );
    m_chPosMEG.reset( new ChannelsPositionsT );
    m_scaleFactors.reset( new std::vector< float > );

    WReaderNeuromagHeader::SPtr reader( new WReaderNeuromagHeader( data, size ) );

    if( !reader->read( m_data.get() ) )
    {
        wlog::error( CLASS ) << "Neuromag header file could not read.";
        return false;
    }

    //std::list< size_t > list;

    m_scaleFactors->clear();
    m_scaleFactors->reserve( m_data->nchan );

    //
    //  Process channel information.
    //
    for( int i = 0; i < m_data->chs.size(); ++i )
    {
        FIFFLIB::FiffChInfo info = m_data->chs.at( i );
        WLEModality::Enum modalityType = WLEModality::fromFiffType( info.kind );

        /*
         bool contains = std::find( list.begin(), list.end(), info.kind ) != list.end();
         if( !contains )
         {
         list.push_back( info.kind );
         }
         */

        //
        // Create pick vectors for all channel types.
        //
        WLEMDRaw::ChanPicksT *vector;
        if( info.kind == WLFiffLib::ChType::STIM ) // skip stimulus channels
        {
            vector = m_stimulusPicks.get();
        }
        else
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
        }

        //
        // Scale factors.
        //
        if( info.range != 0 && info.cal != 0 )
        {
            m_scaleFactors->push_back( info.range * info.cal );
        }
    }

    return true;
}
