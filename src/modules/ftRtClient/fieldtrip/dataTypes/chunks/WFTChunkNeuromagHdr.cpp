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

#include <fstream>

#include "core/dataFormat/fiff/WLFiffChType.h"

#include <core/common/WLogger.h>

#include "core/container/WLArrayList.h"
#include "core/data/enum/WLEModality.h"

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
                WFTAChunk( data, size )
{
    processData( data, size );
}

WLEFTChunkType::Enum WFTChunkNeuromagHdr::getType() const
{
    return WLEFTChunkType::FT_CHUNK_NEUROMAG_HEADER;
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

bool WFTChunkNeuromagHdr::process( const char* data, size_t size )
{
    m_data.reset( new FIFFLIB::FiffInfo );
    m_modalityPicks.reset( new ModalityPicksT );
    m_stimulusPicks.reset( new WLEMDRaw::ChanPicksT );

    std::fstream fostr;
    fostr.open( TMPFILENAME.c_str(), std::fstream::out );

    if( !fostr.is_open() )
    {
        wlog::error( CLASS ) << "Neuromag Header file could not opened.";
        return false;
    }

    fostr.write( data, size );
    fostr.close();

    WReaderNeuromagHeader::SPtr reader( new WReaderNeuromagHeader( TMPFILENAME ) );

    if( !reader->read( m_data.get() ) )
    {
        wlog::error( CLASS ) << "Neuromag header file could not read.";
        return false;
    }

    //
    //  Create pick vectors for all channel types.
    //
    for( int i = 0; i < m_data->chs.size(); ++i )
    {
        FIFFLIB::FiffChInfo info = m_data->chs.at( i );

        WLEMDRaw::ChanPicksT *vector;

        // skip stimulus channels
        if( info.kind == WLFiffLib::ChType::STIM )
        {
            vector = m_stimulusPicks.get();
        }
        else
        {
            WLEModality::Enum modalityType = WLEModality::fromFiffType( info.kind );

            if( modalityType == WLEModality::UNKNOWN )
            {
                continue;
            }

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
    }

    return true;
}
