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

#include "dataTypes/WFTObject.h"

#include "WFTChunkIterator.h"

WFTChunkIterator::WFTChunkIterator( SimpleStorage& buf, int size ) :
                m_store( buf ), m_size( size )
{
    m_pos = 0;
}

bool WFTChunkIterator::hasNext()
{
    return m_pos + sizeof(WFTObject::WFTChunkDefT) < m_size;
}

void WFTChunkIterator::reset()
{
    m_pos = 0;
}

WFTChunk::SPtr WFTChunkIterator::getNext()
{
    if( !hasNext() ) // when end arrived at the end, return an empty pointer.
    {
        return WFTChunk::SPtr();
    }

    WFTObject::WFTChunkDefT *chunkdef = ( WFTObject::WFTChunkDefT * )( ( char * )m_store.data() + m_pos );

    // when the chunk has no data, set position to next chunk and return. (should just happens in case of an error)
    if( chunkdef->size == 0 )
    {
        m_pos += sizeof(WFTObject::WFTChunkDefT);
        return WFTChunk::SPtr();
    }

    const void *srcBuf = chunkdef + sizeof(WFTObject::WFTChunkDefT); // pointer to the chunks data.
    m_pos += sizeof(WFTObject::WFTChunkDefT) + chunkdef->size; // start position for the next chunk (next definition).

    return WFTChunk::SPtr( new WFTChunk( chunkdef->type, chunkdef->size, srcBuf ) );
}
