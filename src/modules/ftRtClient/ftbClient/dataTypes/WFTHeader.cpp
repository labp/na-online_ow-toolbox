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

#include <string>

#include <boost/foreach.hpp>
#include <boost/pointer_cast.hpp>

#include <core/common/WLogger.h>

#include "../container/WFTChunkIterator.h"
#include "../request/WFTRequest_PutHeader.h"

#include "WFTHeader.h"

const std::string WFTHeader::CLASS = "WFTHeader";

WFTHeader::WFTHeader()
{
    init( 0, 0, 0 );
}

WFTHeader::WFTHeader( wftb::nchans_t numChannels, wftb::data_type_t dataType, wftb::fsamp_t fsample )
{
    init( numChannels, dataType, fsample );
}

WFTHeader::~WFTHeader()
{
}

void WFTHeader::init( wftb::nchans_t numChannels, wftb::data_type_t dataType, wftb::fsamp_t fsample )
{
    m_def.nchans = numChannels;
    m_def.data_type = dataType;
    m_def.fsample = fsample;
    m_def.nsamples = 0;
    m_def.bufsize = 0;

    m_chunks = WLList< WFTChunk::SPtr >::instance();
}

WFTRequest::SPtr WFTHeader::asRequest()
{
    if( m_def.nchans == 0 || m_def.fsample == 0 )
    {
        return WFTRequest::SPtr();
    }

    WFTRequest_PutHeader::SPtr request( new WFTRequest_PutHeader( m_def.nchans, m_def.data_type, m_def.fsample ) );

    // add chunks from the collection to the request object.
    BOOST_FOREACH( WFTChunk::SPtr chunk, *m_chunks ){
    request->addChunk( chunk );
}

    return request;
}

bool WFTHeader::parseResponse( WFTResponse::SPtr response )
{
    wlog::debug( CLASS ) << __func__ << "() called.";

    SimpleStorage chunkBuffer; // buffer containing only the chunk data after retrieving.

    if( !response->checkGetHeader( m_def, &chunkBuffer ) )
    {
        return false;
    }

    m_chunks->clear();

    if( m_def.bufsize > 0 )
    {
        // extracts the chunks from the response using an iterator and stores them in the local chunk collection.
        WFTChunkIterator::SPtr iterator( new WFTChunkIterator( &chunkBuffer, m_def.bufsize ) );
        while( iterator->hasNext() )
        {
            m_chunks->push_back( iterator->getNext() );
        }
    }

    wlog::debug( CLASS ) << "Chunks read.";

    return true;
}

wftb::bufsize_t WFTHeader::getSize() const
{
    return sizeof(wftb::HeaderDefT) + m_def.bufsize;
}

wftb::HeaderDefT& WFTHeader::getHeaderDef()
{
    return m_def;
}

wftb::HeaderDefT WFTHeader::getHeaderDef() const
{
    return m_def;
}

bool WFTHeader::hasChunks() const
{
    return m_chunks != 0 && m_chunks->size() > 0;
}

bool WFTHeader::hasChunk( wftb::chunk_type_t chunkType ) const
{
    if( !hasChunks() )
    {
        return false;
    }

    BOOST_FOREACH( WFTChunk::SPtr chunk, *m_chunks ){
    if( chunk->getChunkType() == chunkType )
    return true;
}

    return false;
}

void WFTHeader::addChunk( WFTChunk::SPtr chunk )
{
    m_chunks->push_back( chunk );

    m_def.bufsize += chunk->getDataSize();
}

WLList< WFTChunk::SPtr >::ConstSPtr WFTHeader::getChunks() const
{
    return m_chunks;
}
