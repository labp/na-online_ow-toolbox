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

#include "dataTypes/chunks/WFTAChunkFactory.h"
#include "dataTypes/WFTObject.h"
#include "WFTChunkIterator.h"

WFTChunkIterator::WFTChunkIterator( SimpleStorage& buf, int size ) :
                WFTAIterator< WFTAChunk >( buf, size )
{
}

bool WFTChunkIterator::hasNext() const
{
    return m_pos + ( int )sizeof(WFTChunkDefT) < m_size;
}

WFTAChunk::SPtr WFTChunkIterator::getNext()
{
    if( !hasNext() ) // when end arrived at the end, return an empty pointer.
    {
        return WFTAChunk::SPtr();
    }

    WFTChunkDefT *chunkdef = ( WFTChunkDefT * )( ( char * )m_store.data() + m_pos );

    // when the chunk has no data, set position to next chunk and return. (should just happens in case of an error)
    if( chunkdef->size == 0 )
    {
        m_pos += sizeof(WFTChunkDefT);
        return WFTAChunk::SPtr();
    }

    char *src = ( ( char * )m_store.data() ) + m_pos + sizeof(WFTChunkDefT); // pointer to the chunks data.
    m_pos += sizeof(WFTChunkDefT) + chunkdef->size; // start position for the next chunk (next definition).

    return WFTAChunkFactory< WLEFTChunkType::Enum, WFTAChunk >::create( ( WLEFTChunkType::Enum )chunkdef->type, src,
                    chunkdef->size );
}
