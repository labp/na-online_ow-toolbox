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

#include "../WFTChunkIterator.h"
#include "../WFTRequestBuilder.h"
#include "../request/WFTRequest_PutHeader.h"

#include "WFTHeader.h"

WFTHeader::WFTHeader()
{

}

WFTHeader::WFTHeader( UINT32_T numChannels, UINT32_T dataType, float fsample )
{
    m_def.nchans = numChannels;
    m_def.data_type = dataType;
    m_def.fsample = fsample;
}

WFTHeader::~WFTHeader()
{

}

WFTRequest::SPtr WFTHeader::asRequest()
{
    WFTRequestBuilder::SPtr builder;
    WFTRequest_PutHeader::SPtr request = builder->buildRequest_PUT_HDR( m_def.nchans, m_def.data_type, m_def.fsample );

    // add chunks from the collection to the request object.
    BOOST_FOREACH(WFTChunk::SPtr chunk, m_chunks)
    {
        boost::static_pointer_cast< WFTRequest_PutHeader >( request )->addChunk( chunk );
    }

    return WFTRequest_PutHeader::SPtr( request );
}

bool WFTHeader::parseResponse( WFTResponse::SPtr response )
{
    SimpleStorage chunkBuffer; // buffer containing only the chunk data after retrieving.

    if( !response->checkGetHeader( m_def, &chunkBuffer ) )
    {
        return false;
    }

    // extracts the chunks from the response using an iterator and stores them in the local chunk collection.
    WFTChunkIterator::SPtr iterator( new WFTChunkIterator( chunkBuffer, m_def.bufsize ) );
    while( iterator->hasNext() )
    {
        m_chunks.push_back( iterator->getNext() );
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

bool WFTHeader::hasChunks()
{
    return m_def.bufsize > 0;
}

void WFTHeader::addChunk( WFTChunk::SPtr chunk )
{
    m_chunks.push_back( chunk );

    m_def.bufsize += chunk->getDef().size + sizeof(WFTObject::WFTChunkDefT);
}

WFTHeader::WFTChunkList_ConstSPtr WFTHeader::getChunks()
{
    return WFTHeader::WFTChunkList_ConstSPtr( new WFTChunkList( m_chunks ) );
}
