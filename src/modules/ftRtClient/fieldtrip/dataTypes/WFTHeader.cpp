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

#include <boost/foreach.hpp>
#include <boost/pointer_cast.hpp>

#include "modules/ftRtClient/fieldtrip/WFTChunkIterator.h"
#include "modules/ftRtClient/fieldtrip/io/request/WFTRequest_PutHeader.h"

#include "chunks/WFTChunkFactory.h"
#include "WFTHeader.h"

const std::string WFTHeader::CLASS = "WFTHeader";

WFTHeader::WFTHeader()
{
    init( 0, 0, 0 );
}

WFTHeader::WFTHeader( UINT32_T numChannels, UINT32_T dataType, float fsample )
{
    init( numChannels, dataType, fsample );
}

void WFTHeader::init( UINT32_T numChannels, UINT32_T dataType, float fsample )
{
    m_def.nchans = numChannels;
    m_def.data_type = dataType;
    m_def.fsample = fsample;
    m_def.nsamples = 0;
    m_def.bufsize = 0;

    m_chunks.reset( new WFTChunkList );

    m_channelNames.reset( new WFTChannelNames );
    /*
     = boost::dynamic_pointer_cast< WFTChannelNames >(
     WFTAChunkFactory< WLEFTChunkType::Enum, WFTChunk >::create( WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES ) );
     */
}

WFTRequest::SPtr WFTHeader::asRequest()
{
    if( m_def.nchans == 0 || m_def.fsample == 0 )
    {
        return WFTRequest::SPtr();
    }

    WFTRequest_PutHeader::SPtr request( new WFTRequest_PutHeader( m_def.nchans, m_def.data_type, m_def.fsample ) );

    // add chunks from the collection to the request object.
    BOOST_FOREACH(WFTChunk::SPtr chunk, *m_chunks)
    {
        request->addChunk( chunk );
    }

    return request;
}

bool WFTHeader::parseResponse( WFTResponse::SPtr response )
{
    SimpleStorage chunkBuffer; // buffer containing only the chunk data after retrieving.

    if( !response->checkGetHeader( m_def, &chunkBuffer ) )
    {
        return false;
    }

    m_chunks->clear();
    // extracts the chunks from the response using an iterator and stores them in the local chunk collection.
    WFTChunkIterator::SPtr iterator( new WFTChunkIterator( chunkBuffer, m_def.bufsize ) );
    while( iterator->hasNext() )
    {
        m_chunks->push_back( iterator->getNext() );
    }

    // todo(maschke): chunk processing at this place
    if( this->hasChunks() )
    {

    }

    return true;
}

UINT32_T WFTHeader::getSize() const
{
    return sizeof(WFTHeaderDefT) + m_def.bufsize;
}

WFTHeader::WFTHeaderDefT& WFTHeader::getHeaderDef()
{
    return m_def;
}

WFTHeader::WFTHeaderDefT WFTHeader::getHeaderDef() const
{
    return m_def;
}

WFTChannelNames::SPtr WFTHeader::channelNames() const
{
    return m_channelNames;
}

bool WFTHeader::hasChunks() const
{
    return m_def.bufsize > 0;
}

bool WFTHeader::hasChunk( WLEFTChunkType::Enum chunkType ) const
{
    if( !hasChunks() )
    {
        return false;
    }

    BOOST_FOREACH(WFTChunk::SPtr chunk, *m_chunks)
    {
        if( chunk->getType() == chunkType )
            return true;
    }

    return false;
}

void WFTHeader::addChunk( WFTChunk::SPtr chunk )
{
    m_chunks->push_back( chunk );

    m_def.bufsize += chunk->getSize();
}

WFTChunkList::ConstSPtr WFTHeader::getChunks() const
{
    return m_chunks;
}

WFTChunkList::SPtr WFTHeader::getChunks( WLEFTChunkType::Enum chunkType )
{
    return m_chunks->filter( chunkType );
}

WFTHeader::MeasurementInfo_SPtr WFTHeader::getMeasurementInfo()
{
    return m_measurementInfo;
}

WLList< WLDigPoint >::SPtr WFTHeader::getDigPoints()
{
    return m_digPoints;
}

void WFTHeader::setMeasurementInfo( MeasurementInfo_SPtr info )
{
    m_measurementInfo = info;
}

void WFTHeader::setDigPoints( WLList< WLDigPoint >::SPtr digPoints )
{
    m_digPoints = digPoints;
}
