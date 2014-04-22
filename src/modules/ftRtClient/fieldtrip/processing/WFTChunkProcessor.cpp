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
#include <set>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/foreach.hpp>

#include <QtCore/qstring.h>

#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDEOG.h"
#include "core/data/emd/WLEMDMEG.h"

#include "core/common/WLogger.h"

#include "modules/ftRtClient/fieldtrip/dataTypes/enum/WLEFTChunkType.h"
#include "modules/ftRtClient/reader/WReaderNeuromagHeader.h"

// todo(maschke): remove unused includes

#include "WFTChunkProcessor.h"

const std::string WFTChunkProcessor::CLASS = "WFTChunkProcessor";

#ifdef _WIN32

const std::string WFTChunkProcessor::TMPDIRPATH = "C:/Windows/temp/";

#else

const std::string WFTChunkProcessor::TMPDIRPATH = "/tmp/";

#endif

const std::string WFTChunkProcessor::TMPFILENAME = TMPDIRPATH + "neuromag_header.fif";

WFTChunkProcessor::~WFTChunkProcessor()
{
}

WLArrayList< std::string >::SPtr WFTChunkProcessor::extractChannelNames( WFTChunkList::SPtr chunkList )
{
    WLArrayList< std::string >::SPtr vector( new WLArrayList< std::string >() );

    BOOST_FOREACH(WFTChunk::SPtr chunk, *(chunkList->filter(WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES)))
    {
        channelNamesChunk( chunk, vector );
    }

    return vector;
}

bool WFTChunkProcessor::processNeuromagHeader( WFTChunk::SPtr chunk )
{
    if( chunk->getType() != WLEFTChunkType::FT_CHUNK_NEUROMAG_HEADER )
    {
        return false;
    }

    std::fstream fostr;
    m_measInfo.reset( new FIFFLIB::FiffInfo );
    fostr.open( TMPFILENAME.c_str(), std::fstream::out );

    if( !fostr.is_open() )
    {
        wlog::error( CLASS ) << "Neuromag Header file could not opened.";
        return false;
    }

    fostr.write( ( const char* )chunk->getData(), chunk->getDef().size );
    fostr.close();

    WReaderNeuromagHeader::SPtr reader( new WReaderNeuromagHeader( TMPFILENAME ) );
    if( !reader->read( m_measInfo.get() ) )
    {
        wlog::error( CLASS ) << "Neuromag header file was not read.";
        return false;
    }

    return true;
}

WFTHeader::MeasurementInfo_SPtr WFTChunkProcessor::getMeasurementInfo()
{
    return m_measInfo;
}

bool WFTChunkProcessor::hasMeasurementInfo()
{
    return m_measInfo != 0;
}

bool WFTChunkProcessor::channelNamesChunk( WFTChunk::SPtr chunk, WLArrayList< std::string >::SPtr& names )
{
    names.reset( new WLArrayList< std::string > );

    if( chunk->getType() != WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES )
    {
        return false;
    }

    std::vector< std::string > splitVec;
    std::string str = chunk->getDataString();

    boost::split( splitVec, str, boost::is_any_of( "\0" ), boost::token_compress_on );

    if( splitVec.size() == 0 )
    {
        return false;
    }

    BOOST_FOREACH(std::string chanName, splitVec)
    {
        names->push_back( chanName );
    }

    return names->size() > 0;
}

bool WFTChunkProcessor::channelNamesMeasInfo( WLArrayList< std::string >::SPtr& names )
{
    names.reset( new WLArrayList< std::string > );

    if(m_measInfo == 0)
    {
        return false;
    }

    for( int i = 0; i < m_measInfo->chs.size(); ++i )
    {
        names->push_back( m_measInfo->chs.at( i ).ch_name.toStdString() );
    }

    return true;
}

bool WFTChunkProcessor::channelNamesMeasInfo( WFTChunk::SPtr chunk, WLArrayList< std::string >::SPtr& names )
{
    if( !chunk->getType() == WLEFTChunkType::FT_CHUNK_NEUROMAG_HEADER )
    {
        return false;
    }

    if( !processNeuromagHeader( chunk ) )
    {
        return false;
    }

    return channelNamesMeasInfo( names );
}

/*
 WLArrayList< std::string >::SPtr WFTChunkProcessor::extractChannelFlags( WFTChunkList::SPtr chunkList )
 {
 //boost::shared_ptr< std::set< WLEModality::Enum > > modalities( new std::set< WLEModality::Enum > );

 WLArrayList< std::string >::SPtr vector( new WLArrayList< std::string >() );

 BOOST_FOREACH(WFTChunk::SPtr chunk, *(chunkList->filter(WLEFTChunkType::FT_CHUNK_CHANNEL_FLAGS)))
 {
 //getChannelModality( chunk, *modalities );

 vector->push_back( chunk->getDataString() );
 }

 return vector;
 }
 */

/*
 bool WFTChunkProcessor::getChannelNames( WFTChunk::SPtr chunk, WLArrayList< std::string >::SPtr &vector )
 {
 if( chunk->getType() != WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES )
 return false;

 std::vector< std::string > splitVec;
 std::string str = chunk->getDataString();

 boost::split( splitVec, str, boost::is_any_of( "\0" ), boost::token_compress_on );

 if( splitVec.size() == 0 )
 {
 return false;
 }

 BOOST_FOREACH(std::string chanName, splitVec)
 {
 vector->push_back( chanName );
 }

 return vector->size() > 0;
 }
 */
